"""Step 1 -- does the genotype surrogate transfer?

Three transfers, each one a condition a practitioner faces before any label of their own:

  dataset  : fit on CIFAR-10, rank CIFAR-100 (and the reverse) with *no* target label;
             few-shot = the same model plus a ridge on the residual of n target labels;
  arm      : fit on the baseline arm alone, rank the architectures the LLM arms sampled --
             a distribution shift, since the arms search differently;
  reference: fit against the 1-epoch accuracy, rank the 3-epoch one -- a fidelity shift.

Writes results/transfer.json, tables/tab_transfer.tex, figures/fig_transfer.pdf.
"""
import json, os
import numpy as np
import common, surrogate as sg
from scipy.stats import spearmanr

NS = [0, 25, 50, 100, 250, 500]
REPS = 5
HOLD = 1200


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def topk(sig, y, k):
    return len(set(np.argsort(-sig)[:k].tolist()) & set(np.argsort(-y)[:k].tolist())) / k


def load(ds):
    rows = [r for r in common.gene_table() if r['dataset'] == ds and r['acc3'] is not None]
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows])
    a1 = np.array([r['acc1'] if r['acc1'] is not None else np.nan for r in rows])
    arm = np.array([r['method'] for r in rows])
    return P, y, a1, arm


def few_shot(Psrc, ysrc, Ptgt, ytgt, te, pool):
    """Source model, then a ridge on the target residual of n labels. Target-only is the control."""
    msrc = sg.PairwiseRidge(10.0).fit(Psrc, ysrc)
    base = msrc.predict(Ptgt[te])
    out = {}
    for n in NS:
        tr, ctl = [], []
        for rep in range(REPS):
            rg = np.random.default_rng(1000 + rep)
            if n == 0:
                tr.append(rho(base, ytgt[te]))
                continue
            sub = rg.choice(pool, n, replace=False)
            res = ytgt[sub] - msrc.predict(Ptgt[sub])
            corr = sg.PairwiseRidge(100.0).fit(Ptgt[sub], res)
            tr.append(rho(base + corr.predict(Ptgt[te]), ytgt[te]))
            ctl.append(rho(sg.PairwiseRidge(10.0).fit(Ptgt[sub], ytgt[sub]).predict(Ptgt[te]),
                           ytgt[te]))
        out[n] = {'transfer': float(np.mean(tr)), 'transfer_sd': float(np.std(tr)),
                  'scratch': float(np.mean(ctl)) if ctl else None,
                  'scratch_sd': float(np.std(ctl)) if ctl else None}
    return out


def main():
    D = {ds: load(ds) for ds in common.DATASETS}
    out = {}

    # ---- transfer across datasets -------------------------------------------------
    out['dataset'] = {}
    for src, tgt in (('cifar10', 'cifar100'), ('cifar100', 'cifar10')):
        Ps, ys, _, _ = D[src]
        Pt, yt, a1t, _ = D[tgt]
        rg = np.random.default_rng(0)
        idx = rg.permutation(len(yt))
        te, pool = idx[:HOLD], idx[HOLD:]
        fs = few_shot(Ps, ys, Pt, yt, te, pool)
        okt = ~np.isnan(a1t[te])
        p0 = sg.PairwiseRidge(10.0).fit(Ps, ys).predict(Pt[te])
        out['dataset'][f'{src}->{tgt}'] = {
            'n_src': len(ys), 'n_test': HOLD, 'few_shot': fs,
            'epoch1_on_target': rho(a1t[te][okt], yt[te][okt]),
            'top10': topk(p0, yt[te], 10), 'top50': topk(p0, yt[te], 50),
            'top100': topk(p0, yt[te], 100)}

    # ---- transfer across arms (distribution shift) --------------------------------
    out['arm'] = {}
    for ds in common.DATASETS:
        P, y, a1, arm = D[ds]
        tr = arm == 'baseline'
        for name, mask in (('llm', np.isin(arm, common.LLMS)), ('frozen', arm == 'frozen')):
            m = sg.PairwiseRidge(10.0).fit(P[tr], y[tr])
            ok = ~np.isnan(a1[mask])
            out['arm'].setdefault(ds, {})[f'baseline->{name}'] = {
                'n_fit': int(tr.sum()), 'n_test': int(mask.sum()),
                'rho': rho(m.predict(P[mask]), y[mask]),
                'epoch1': rho(a1[mask][ok], y[mask][ok]),
                'top50': topk(m.predict(P[mask]), y[mask], 50)}

    # ---- transfer across the fidelity of the reference ----------------------------
    out['reference'] = {}
    for ds in common.DATASETS:
        P, y, a1, _ = D[ds]
        ok = ~np.isnan(a1)
        rg = np.random.default_rng(0)
        idx = rg.permutation(int(ok.sum()))
        Pk, yk, ak = P[ok], y[ok], a1[ok]
        te, tr = idx[:HOLD], idx[HOLD:]
        m1 = sg.PairwiseRidge(10.0).fit(Pk[tr], ak[tr])       # fitted on 1-epoch labels
        m3 = sg.PairwiseRidge(10.0).fit(Pk[tr], yk[tr])       # fitted on 3-epoch labels
        out['reference'][ds] = {'fit_on_epoch1_rank_epoch3': rho(m1.predict(Pk[te]), yk[te]),
                                'fit_on_epoch3_rank_epoch3': rho(m3.predict(Pk[te]), yk[te]),
                                'n_test': HOLD}

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(f'{common.RES}/transfer.json', 'w'), indent=1)

    # ---- table --------------------------------------------------------------------
    d = out['dataset']
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Transfer of the genotype surrogate. Spearman $\rho$ against the 3-epoch',
         r"reference on a held-out set the model never saw. ``Fitted here'' is the same model",
         r'trained on target labels only; the transferred model with no target label at all',
         r'already outranks it with 500.}', r'\label{tab:transfer}',
         r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{llcc}', r'\toprule',
         r'Transfer & Fitted on & $\rho$ & Reference \\', r'\midrule']
    for k in ('cifar10->cifar100', 'cifar100->cifar10'):
        e = d[k]
        s, t = k.split('->')
        sl, tl = s.upper().replace('CIFAR', 'CIFAR-'), t.upper().replace('CIFAR', 'CIFAR-')
        L.append(rf'\multirow{{3}}{{*}}{{{sl} $\to$ {tl}}} & {sl}, 0 target labels & '
                 rf'{e["few_shot"][0]["transfer"]:.3f} & --- \\')
        L.append(rf' & {sl} + 25 target labels & {e["few_shot"][25]["transfer"]:.3f} & '
                 rf'{e["few_shot"][25]["scratch"]:.3f} \\')
        L.append(rf' & {sl} + 500 target labels & {e["few_shot"][500]["transfer"]:.3f} & '
                 rf'{e["few_shot"][500]["scratch"]:.3f} \\')
        L.append(rf'\cmidrule(l){{2-4}}')
    for ds in common.DATASETS:
        a = out['arm'][ds]['baseline->llm']
        dl = ds.upper().replace('CIFAR', 'CIFAR-')
        L.append(rf'Baseline arm $\to$ LLM arms & {dl}, {a["n_fit"]} archs & {a["rho"]:.3f} & '
                 rf'{a["epoch1"]:.3f} \\')
    for ds in common.DATASETS:
        r_ = out['reference'][ds]
        dl = ds.upper().replace('CIFAR', 'CIFAR-')
        L.append(rf'1-epoch labels $\to$ 3-epoch rank & {dl} & '
                 rf'{r_["fit_on_epoch1_rank_epoch3"]:.3f} & '
                 rf'{r_["fit_on_epoch3_rank_epoch3"]:.3f} \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_transfer.tex', 'w').write('\n'.join(L) + '\n')

    # ---- report -------------------------------------------------------------------
    print('== dataset transfer ==')
    for k, e in out['dataset'].items():
        print(f'  {k}: 0 labels rho={e["few_shot"][0]["transfer"]:.3f} | '
              f'scratch@500={e["few_shot"][500]["scratch"]:.3f} | '
              f'1 epoch on target={e["epoch1_on_target"]:.3f} | '
              f'top50 recall={e["top50"]:.2f}')
    print('== arm transfer (distribution shift) ==')
    for ds, e in out['arm'].items():
        for k, v in e.items():
            print(f'  {ds} {k}: fit n={v["n_fit"]} -> rho={v["rho"]:.3f} '
                  f'(1 epoch on the same archs {v["epoch1"]:.3f})')
    print('== reference transfer ==')
    for ds, v in out['reference'].items():
        print(f'  {ds}: fit on 1-epoch labels -> rank 3-epoch rho='
              f'{v["fit_on_epoch1_rank_epoch3"]:.3f} (fit on 3-epoch {v["fit_on_epoch3_rank_epoch3"]:.3f})')


if __name__ == '__main__':
    main()
