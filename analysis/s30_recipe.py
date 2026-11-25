"""Step 10 -- the whole recipe, end to end, on measured architectures.

The three results of Sections IV-VI compose into one policy. At a fixed budget of 100 training
epochs, starting from a pool of real architectures whose 1-, 2- and 3-epoch accuracies were all
measured:

  A  train 100 candidates for one epoch, return the best            (what the pipeline does)
  B  screen 800 free, train the best 100 for one epoch              (Section V as published)
  C  screen 800 free, train 73, promote the top quarter to a second
     epoch and half of those to a third                             (+ successive halving)
  D  as C, but the promotion ranks on the free surrogate and the
     measured epoch fused 2:1                                       (+ the residual result)

Every policy costs exactly 100 epochs and every reported outcome is the measured 3-epoch
accuracy of the architecture the policy returned. The surrogate is fitted on a disjoint half of
the architectures, split by a hash of the uid, exactly as in s15_prescreen.py.

Writes results/recipe.json, tables/tab_recipe.tex.
"""
import csv, hashlib, json, os
import numpy as np
import common, surrogate as sg
from scipy.stats import rankdata

RNG = np.random.default_rng(20261009)
T = 4000
BUDGET = 100


def half(uid):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2


def boot_ci(v, n=2000):
    v = np.asarray(v, float)
    m = RNG.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def three_epoch_curves():
    by = {}
    for rec in csv.DictReader(open(os.path.join(common.DATA, 'learning_curves.csv'))):
        by.setdefault((rec['uid'], rec['dataset']), {})[int(rec['epoch'])] = \
            float(rec['test_acc'])
    return {k: v for k, v in by.items() if {1, 2, 3} <= set(v)}


def main():
    cur = three_epoch_curves()
    out = {}
    for ds in common.DATASETS:
        rows = [r for r in common.gene_table()
                if r['dataset'] == ds and (r['uid'], ds) in cur]
        fit = [r for r in rows if half(r['uid']) == 0]
        pool = [r for r in rows if half(r['uid']) == 1]
        model = sg.PairwiseRidge(10.0).fit(
            np.array([sg.codes_of(r) for r in fit]),
            np.array([r['acc3'] for r in fit]))
        s = model.predict(np.array([sg.codes_of(r) for r in pool]))
        e1 = np.array([cur[(r['uid'], ds)][1] for r in pool])
        e2 = np.array([cur[(r['uid'], ds)][2] for r in pool])
        e3 = np.array([cur[(r['uid'], ds)][3] for r in pool])
        n = len(pool)
        rs = rankdata(s) / n
        cells = {}

        def rec(name, got, cost, note=''):
            lo, hi = boot_ci(got)
            cells[name] = dict(acc=float(np.mean(got)), lo=lo, hi=hi, epochs=int(cost),
                               note=note)

        def draws(m):
            return RNG.integers(0, n, (T, m))

        # --- A: train 100 for one epoch, select on the measured epoch
        d = draws(BUDGET)
        rec('A_train100', e3[d[np.arange(T), e1[d].argmax(1)]], BUDGET,
            'train 100 for one epoch')

        # --- B: screen 800 free, train the best 100 for one epoch
        d = draws(800)
        keep = np.take_along_axis(d, np.argsort(-s[d], 1)[:, :BUDGET], 1)
        rec('B_screen800', e3[keep[np.arange(T), e1[keep].argmax(1)]], BUDGET,
            'screen 800, train 100')

        # --- C/D: screen 800 free, train N1, halve twice, within the same 100 epochs
        def halving(fuse):
            n1 = 73
            n2, n3 = 18, 9                       # 73 + 18 + 9 = 100 epochs
            d = draws(800)
            k1 = np.take_along_axis(d, np.argsort(-s[d], 1)[:, :n1], 1)
            if fuse:
                score1 = 2 * rankdata(e1[k1], axis=1) / n1 + rs[k1]
            else:
                score1 = e1[k1]
            k2 = np.take_along_axis(k1, np.argsort(-score1, 1)[:, :n2], 1)
            k3 = np.take_along_axis(k2, np.argsort(-e2[k2], 1)[:, :n3], 1)
            return e3[k3[np.arange(T), e3[k3].argmax(1)]], n1 + n2 + n3

        g, c = halving(False)
        rec('C_screen_halve', g, c, 'screen 800, 73 x1ep, 18 x2ep, 9 x3ep')
        g, c = halving(True)
        rec('D_screen_halve_fused', g, c, 'as C, promotion fused with the surrogate 2:1')

        # --- reference points
        d = draws(BUDGET // 3)
        rec('ref_three_epochs_33', e3[d].max(1), 3 * (BUDGET // 3),
            'three epochs on 33 candidates')
        d = draws(800)
        rec('ref_surrogate_only', e3[d[np.arange(T), s[d].argmax(1)]], 0,
            'surrogate alone, no training')
        out[ds] = dict(n_fit=len(fit), n_pool=len(pool), cells=cells)
    json.dump(out, open(f'{common.RES}/recipe.json', 'w'), indent=1)

    ORDER = [('A_train100', 'Train 100 for one epoch (flat, the usual policy)'),
             ('ref_surrogate_only', 'Screen 800 free, train none'),
             ('B_screen800', r'\quad $+$ one epoch on the best 100'),
             ('C_screen_halve', r'\quad $+$ successive halving')]
    L = [r'\begin{table}[!htb]', r'\centering',
         r'\caption{The whole recipe at one budget. Every policy that trains anything spends',
         r'100 training epochs (99 for the full-fidelity reference, which cannot divide',
         r'evenly), on architectures whose 1-, 2- and 3-epoch accuracies were all measured. The',
         r'number reported is the measured 3-epoch accuracy of the architecture the policy',
         r'returned, over 4{,}000 draws, with a 95\% bootstrap interval. The first line is the',
         r'flat low-fidelity policy a NAS pipeline usually uses: one epoch on every candidate,',
         r'keep the best; the next three start from free screening and each adds one component',
         r'of the recipe to the line above. Differences are computed before rounding. Our own',
         r'pipeline selected on an LLM predictor that needs three epochs, which is worse still',
         r'(the budget section of the main paper).}',
         r'\label{tab:recipe}',
         r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lcc}', r'\toprule',
         r'Policy (100 training epochs) & CIFAR-10 & CIFAR-100 \\', r'\midrule']
    for key, name in ORDER:
        a, b = out['cifar10']['cells'][key], out['cifar100']['cells'][key]
        L.append(rf'{name} & {a["acc"]:.2f} & {b["acc"]:.2f} \\')
    a0 = out['cifar10']['cells']
    b0 = out['cifar100']['cells']
    L += [r'\midrule',
          rf'\textbf{{Full recipe (screen $+$ halving) $-$ as run}} & '
          rf'\textbf{{{a0["C_screen_halve"]["acc"] - a0["A_train100"]["acc"]:+.2f}}} & '
          rf'\textbf{{{b0["C_screen_halve"]["acc"] - b0["A_train100"]["acc"]:+.2f}}} \\',
          r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_recipe.tex', 'w').write('\n'.join(L) + '\n')

    for ds, o in out.items():
        base = o['cells']['A_train100']['acc']
        print(f"== {ds} (fit {o['n_fit']}, pool {o['n_pool']}) ==")
        for k, c in o['cells'].items():
            print(f"   {k:22s} {c['epochs']:4d} ep  {c['acc']:6.2f} "
                  f"[{c['lo']:.2f},{c['hi']:.2f}]  {c['acc']-base:+.2f}")


if __name__ == '__main__':
    main()
