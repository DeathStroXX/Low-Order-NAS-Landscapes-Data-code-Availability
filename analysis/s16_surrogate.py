"""Step 2 -- the zero-cost genotype surrogate: fidelity, sample efficiency, cross-run transfer.

A ridge on the 13 genes' indicators plus all gene-pair interactions (surrogate.py) costs no
training epoch and no LLM call. This script measures it against every other signal the
pipeline had available, asks how many labelled architectures it needs, and checks that it
transfers to a run it never saw -- the condition a search must satisfy to use it online.

Writes results/surrogate.json, tables/tab_surrogate.tex, data/surrogate_predictions.csv.
"""
import json, os, csv
import numpy as np
import common, surrogate as sg
from scipy.stats import spearmanr

RNG = np.random.default_rng(7)
FOLDS = 5


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def topk(sig, y, k=10):
    return len(set(np.argsort(-sig)[:k].tolist()) & set(np.argsort(-y)[:k].tolist())) / k


def main():
    table = [r for r in common.gene_table()
             if r['acc1'] is not None and r['acc3'] is not None]
    out, dump = {}, []
    for ds in common.DATASETS:
        rows = [r for r in table if r['dataset'] == ds]
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        a1 = np.array([r['acc1'] for r in rows])
        pr = np.array([r['pred'] if r['pred'] is not None else np.nan for r in rows])
        ok = ~np.isnan(pr)
        n = len(y)
        gf = sg.GramFitter(P, y)
        idx = np.random.default_rng(0).permutation(n)

        # --- cross-validated fidelity of the surrogate
        p_sur = np.zeros(n)
        for f in range(FOLDS):
            te = idx[f::FOLDS]
            keep = np.ones(n, bool); keep[te] = False
            p_sur[te] = gf.fit_mask(keep)(te)
        # --- surrogate + one measured epoch (the epoch enters as a linear residual term)
        p_comb = np.zeros(n)
        for f in range(FOLDS):
            te = idx[f::FOLDS]
            keep = np.ones(n, bool); keep[te] = False
            pred = gf.fit_mask(keep)
            base_tr = pred(np.flatnonzero(keep))
            co = np.polyfit(a1[keep], y[keep] - base_tr, 1)
            p_comb[te] = pred(te) + np.polyval(co, a1[te])

        # --- a nonparametric alternative on the identical features and folds, so that the
        #     choice of a ridge over the gene pairs is a measured choice and not an assertion
        from sklearn.ensemble import RandomForestRegressor
        X = sg.design(P)
        p_rf = np.zeros(n)
        for f in range(FOLDS):
            te = idx[f::FOLDS]
            keep = np.ones(n, bool); keep[te] = False
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                       random_state=0).fit(X[keep], y[keep])
            p_rf[te] = rf.predict(X[te])

        res = {'n': n, 'n_features': sg.NFEAT,
               'rho': {'predictor_llm': rho(pr[ok], y[ok]),
                       'surrogate': rho(p_sur, y),
                       'epoch1': rho(a1, y),
                       'random_forest': rho(p_rf, y),
                       'surrogate_plus_epoch1': rho(p_comb, y)},
               'top10': {'predictor_llm': topk(np.where(ok, np.nan_to_num(pr, nan=-1e9), -1e9), y),
                         'surrogate': topk(p_sur, y),
                         'epoch1': topk(a1, y),
                         'surrogate_plus_epoch1': topk(p_comb, y)},
               'mae': float(np.abs(p_sur - y).mean())}

        # --- how many labelled architectures the surrogate needs
        eff = {}
        for m in (50, 100, 250, 500, 1000, 2000):
            if m > n - 300:
                continue
            rr = []
            for rep in range(10):
                pm = RNG.permutation(n)
                keep = np.zeros(n, bool); keep[pm[:m]] = True
                te = pm[m:]
                rr.append(rho(gf.fit_mask(keep)(te), y[te]))
            eff[int(m)] = [float(np.mean(rr)), float(np.std(rr))]
        res['sample_efficiency'] = eff

        # --- leave-one-run-out: usable online inside a run it never saw
        loo = {}
        for rn in sorted({r['run'] for r in rows}):
            te = np.flatnonzero(np.array([r['run'] == rn for r in rows]))
            if len(te) < 40:
                continue
            keep = np.ones(n, bool); keep[te] = False
            p = gf.fit_mask(keep)(te)
            loo[rn] = dict(n=int(len(te)), rho=rho(p, y[te]), rho_epoch1=rho(a1[te], y[te]))
            for r, pv in zip([rows[i] for i in te], p):
                dump.append(dict(uid=r['uid'], dataset=ds, run=rn, method=r['method'],
                                 acc_epoch1=r['acc1'], acc_epoch3_true=r['acc3'],
                                 predicted_fitness=r['pred'],
                                 surrogate_loo=round(float(pv), 4)))
        v = [d['rho'] for d in loo.values()]
        res['loo'] = dict(per_run=loo, mean=float(np.mean(v)), sd=float(np.std(v)),
                          min=float(np.min(v)), n_runs=len(v),
                          mean_epoch1=float(np.mean([d['rho_epoch1'] for d in loo.values()])))
        out[ds] = res

    json.dump(out, open(os.path.join(common.RES, 'surrogate.json'), 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    with open(os.path.join(common.HERE, '..', 'data', 'surrogate_predictions.csv'), 'w',
              newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dump[0]), lineterminator='\n')
        w.writeheader(); w.writerows(dump)
    _tex(out)
    for ds, r in out.items():
        print(f"\n== {ds}  n={r['n']}  features={r['n_features']}  MAE={r['mae']:.2f}")
        for k in r['rho']:
            t = r['top10'].get(k)
            tail = '' if t is None else f"   top-10 {t * 100:4.0f}%"
            print(f"   {k:24s} rho {r['rho'][k]:.3f}{tail}")
        print("   architectures needed:",
              {k: round(v[0], 3) for k, v in r['sample_efficiency'].items()})
        L = r['loo']
        print(f"   leave-one-run-out over {L['n_runs']} runs: rho = {L['mean']:.3f}"
              f" +-{L['sd']:.3f} (min {L['min']:.3f});  one epoch on the same runs"
              f" {L['mean_epoch1']:.3f}")
    print(f"\nwrote data/surrogate_predictions.csv ({len(dump)} rows)")


def _tex(out):
    rows = [('predictor_llm', 'LLM accuracy predictor', 'one LLM call, after 3 epochs'),
            ('surrogate', 'Genotype surrogate (13 genes)', 'none'),
            ('epoch1', 'One training epoch', '1 epoch'),
            ('surrogate_plus_epoch1', 'Genotype surrogate $+$ one epoch', '1 epoch')]
    L = ['\\begin{tabular}{l l rr rr}', '\\toprule',
         '& & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
         '\\textbf{Selection signal} & \\textbf{Cost per architecture} & $\\rho$ '
         '& \\textbf{top-10} & $\\rho$ & \\textbf{top-10} \\\\', '\\midrule']
    for k, lab, cost in rows:
        a, b = out['cifar10'], out['cifar100']
        L.append(f"{lab} & {cost} & {a['rho'][k]:.3f} & {a['top10'][k]*100:.0f}\\% & "
                 f"{b['rho'][k]:.3f} & {b['top10'][k]*100:.0f}\\% \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_surrogate.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
