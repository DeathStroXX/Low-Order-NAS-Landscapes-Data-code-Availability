"""Step 1 -- a free genotype surrogate used to pre-screen candidates before training.

The question: at a FIXED training budget of B epochs, is it better to train B candidates for
one epoch each, or to score M > B candidates with a zero-cost genotype surrogate, keep the
best B, and train only those?

The surrogate is a random forest over the 13 genes, so it consumes no training epochs and no
LLM call. To keep the comparison honest it is fitted on a disjoint half of the architectures
(split by a hash of the uid) and every policy is evaluated on the other half, using the
MEASURED 1-epoch and 3-epoch accuracies of real architectures. Nothing here is circular: the
surrogate only ranks, the reported outcome is always a measured 3-epoch accuracy.

Writes results/prescreen.json and tables/tab_prescreen.tex.
"""
import json, os, hashlib
import numpy as np
import common, surrogate as sg

RNG = np.random.default_rng(20261008)
T = 3000          # policy draws per cell
BUDGET = 100      # training epochs available
CATS = ['activation', 'pooling_type', 'conv_type', 'norm_type', 'optimizer_type']


def codes(rows):
    return np.array([sg.codes_of(r) for r in rows])


def half(uid):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2


def boot_ci(v, n=2000):
    v = np.asarray(v, float)
    m = RNG.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    table = [r for r in common.gene_table()
             if r['acc1'] is not None and r['acc3'] is not None]
    out = {}
    for ds in common.DATASETS:
        rows = [r for r in table if r['dataset'] == ds]
        fit = [r for r in rows if half(r['uid']) == 0]
        pool = [r for r in rows if half(r['uid']) == 1]
        model = sg.PairwiseRidge(10.0).fit(codes(fit), np.array([r['acc3'] for r in fit]))
        s = model.predict(codes(pool))          # surrogate score, zero training cost
        a1 = np.array([r['acc1'] for r in pool])
        a3 = np.array([r['acc3'] for r in pool])
        pr = np.array([r['pred'] if r['pred'] is not None else np.nan for r in pool])
        n = len(pool)

        def draws(m):
            return RNG.integers(0, n, (T, m))

        cells = {}

        def rec(name, got, cost, note=''):
            lo, hi = boot_ci(got)
            cells[name] = dict(acc=float(np.mean(got)), lo=lo, hi=hi,
                               epochs=cost, note=note)

        # (a) spend the whole budget on one epoch each, select on the measured epoch
        d = draws(BUDGET)
        rec('train_all', a3[d[np.arange(T), a1[d].argmax(1)]], BUDGET)
        # (b) free surrogate pre-screen of M candidates, train the top BUDGET for one epoch
        for M in (200, 400, 800, 1600):
            d = draws(M)
            keep = np.take_along_axis(d, np.argsort(-s[d], 1)[:, :BUDGET], 1)
            pick = keep[np.arange(T), a1[keep].argmax(1)]
            rec(f'prescreen_{M}', a3[pick], BUDGET, f'surrogate scores {M}')
        # (c) equal-cost three-epoch oracle: a third as many candidates, trained to the reference
        d = draws(BUDGET // 3)
        rec('oracle3_equalcost', a3[d].max(1), 3 * (BUDGET // 3))
        # (d) the three-epoch reference over the same number of candidates (3x the cost)
        d = draws(BUDGET)
        rec('oracle3_fullcost', a3[d].max(1), 3 * BUDGET)
        # (e) surrogate alone, no training at all
        for M in (100, 800):
            d = draws(M)
            rec(f'surrogate_only_{M}', a3[d[np.arange(T), s[d].argmax(1)]], 0,
                f'no training, {M} candidates')
        # (f) the LLM predictor cannot pre-screen (it reads the measured trajectory), but it
        #     can re-score already-trained candidates: does it help there?
        ok = ~np.isnan(pr)
        if ok.sum() > 1000:
            idx = np.flatnonzero(ok)
            d = idx[RNG.integers(0, len(idx), (T, BUDGET))]
            rec('predictor_rescore', a3[d[np.arange(T), pr[d].argmax(1)]], BUDGET,
                'selects on the predictor instead of the measured epoch')
            rec('train_all_sub', a3[d[np.arange(T), a1[d].argmax(1)]], BUDGET,
                'same candidate subset, selects on the measured epoch')
        out[ds] = dict(n_fit=len(fit), n_pool=len(pool), cells=cells,
                       surrogate_rho=float(_rho(s, a3)))
    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'prescreen.json'), 'w'), indent=1)
    _tex(out)
    for ds, o in out.items():
        base = o['cells']['train_all']['acc']
        print(f"\n== {ds}  (surrogate fitted on {o['n_fit']}, evaluated on {o['n_pool']}; "
              f"out-of-sample rho={o['surrogate_rho']:.3f})")
        for k, c in o['cells'].items():
            print(f"   {k:24s} {c['epochs']:4d} ep  {c['acc']:6.2f} "
                  f"[{c['lo']:.2f},{c['hi']:.2f}]  {c['acc']-base:+.2f}")


def _rho(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic


def _tex(out):
    rowspec = [('train_all', 'Train 100, select on the measured epoch'),
               ('prescreen_200', 'Pre-screen 200 $\\to$ train 100'),
               ('prescreen_400', 'Pre-screen 400 $\\to$ train 100'),
               ('prescreen_800', 'Pre-screen 800 $\\to$ train 100'),
               ('prescreen_1600', 'Pre-screen 1600 $\\to$ train 100'),
               ('oracle3_equalcost', 'Three epochs on 33 candidates'),
               ('surrogate_only_800', 'Surrogate alone, 800 candidates'),
               ('oracle3_fullcost', 'Three epochs on 100 candidates')]
    L = ['\\begin{tabular}{l r rr rr}', '\\toprule',
         '& & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
         '\\textbf{Selection policy} & \\textbf{Epochs} & \\textbf{Acc.} & $\\Delta$ '
         '& \\textbf{Acc.} & $\\Delta$ \\\\', '\\midrule']
    for key, lab in rowspec:
        c10, c100 = out['cifar10']['cells'], out['cifar100']['cells']
        if key not in c10:
            continue
        b10 = c10['train_all']['acc']; b100 = c100['train_all']['acc']
        d10 = '' if key == 'train_all' else f"{c10[key]['acc']-b10:+.2f}"
        d100 = '' if key == 'train_all' else f"{c100[key]['acc']-b100:+.2f}"
        L.append(f"{lab} & {c10[key]['epochs']} & {c10[key]['acc']:.2f} & {d10} "
                 f"& {c100[key]['acc']:.2f} & {d100} \\\\")
        if key == 'train_all':
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_prescreen.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
