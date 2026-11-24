#!/usr/bin/env python3
"""s11_proxy_fidelity.py -- validates the 1-epoch accuracy as a selection signal, across every
run of every arm, and measures what the LLM accuracy predictor adds on top of it.

Measures, per run and then averaged over runs (the run is the unit of analysis):
  rho          Spearman correlation with the 3-epoch reference
  retention    share of the 3-epoch top-10 that is also in the signal's own top-10
  hit@3        whether the signal's best architecture is among the three best at 3 epochs
and, pooled over architectures:
  AUC          ability to flag the worst quartile of architectures
  incremental  gain in rank-R^2 from adding the predictor to a model that already has the proxy

Writes results/proxy.json and tables/tab_proxy.tex.
"""
import json, os
import numpy as np
from scipy import stats
from common import load_all, DATASETS, METHODS, RES, HERE
from statsutil import ci95

T = os.path.join(HERE, '..', 'tables')
ARMLAB = {'baseline': 'Baseline GA', 'frozen': 'Restarts, frozen ops.', 'llm': 'Restarts, LLM ops.'}


def per_run(sig, a3, k=10):
    rho = stats.spearmanr(sig, a3).statistic
    top_t = set(np.argsort(a3)[-k:]); top_s = set(np.argsort(sig)[-k:])
    rank = np.argsort(np.argsort(-a3))
    b = int(np.argmax(sig))
    return rho, 100 * len(top_t & top_s) / k, bool(rank[b] < 3), bool(rank[b] == 0), 100 * k / len(a3)


def auc(score, lab):
    r = stats.rankdata(score); n1 = int(lab.sum()); n0 = len(lab) - n1
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == '__main__':
    R = load_all()
    out = {}
    for ds in DATASETS:
        groups = {'baseline': R[(ds, 'baseline')], 'frozen': R[(ds, 'frozen')],
                  'llm': [x for m in ['qwen', 'deepseek', 'mistral'] for x in R[(ds, m)]]}
        d = {'arms': {}}
        for g, rs in groups.items():
            acc = {'proxy': [], 'pred': []}
            for name, key in [('proxy', 'acc1'), ('pred', 'pr')]:
                for x in rs:
                    acc[name].append(per_run(x[key], x['acc3']))
            d['arms'][g] = {}
            for name in ('proxy', 'pred'):
                v = acc[name]
                d['arms'][g][name] = dict(n_runs=len(v), rho=ci95([a[0] for a in v]),
                                          retention=ci95([a[1] for a in v]),
                                          hit3=100 * float(np.mean([a[2] for a in v])),
                                          hit1=100 * float(np.mean([a[3] for a in v])),
                                          chance=float(np.mean([a[4] for a in v])))
        allr = [x for rs in groups.values() for x in rs]
        d['all'] = {}
        for name, key in [('proxy', 'acc1'), ('pred', 'pr')]:
            v = [per_run(x[key], x['acc3']) for x in allr]
            d['all'][name] = dict(n_runs=len(v), rho=ci95([a[0] for a in v]),
                                  rho_min=float(min(a[0] for a in v)), rho_max=float(max(a[0] for a in v)),
                                  rho_sd=float(np.std([a[0] for a in v], ddof=1)),
                                  retention=ci95([a[1] for a in v]),
                                  hit3=100 * float(np.mean([a[2] for a in v])),
                                  hit1=100 * float(np.mean([a[3] for a in v])),
                                  chance=float(np.mean([a[4] for a in v])))
        a1 = np.concatenate([x['acc1'] for x in allr]); a3 = np.concatenate([x['acc3'] for x in allr])
        pp = np.concatenate([x['pr'] for x in allr])
        bad = a3 <= np.quantile(a3, 0.25)
        y = stats.rankdata(a3)
        X1 = np.c_[np.ones(len(a1)), stats.rankdata(a1)]
        X2 = np.c_[X1, stats.rankdata(pp)]
        r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        r2 = y - X2 @ np.linalg.lstsq(X2, y, rcond=None)[0]
        d['pooled'] = dict(n=int(len(a3)),
                           auc_proxy=float(1 - auc(a1, bad)), auc_pred=float(1 - auc(pp, bad)),
                           rank_r2_proxy=float(1 - r1.var() / y.var()),
                           rank_r2_both=float(1 - r2.var() / y.var()),
                           incremental=float((1 - r2.var() / y.var()) - (1 - r1.var() / y.var())),
                           pred_distinct=int(len(np.unique(pp))), pred_max=float(pp.max()),
                           pred_mean=float(pp.mean()), obs_max=float(a3.max()))
        out[ds] = d
    json.dump(out, open(os.path.join(RES, 'proxy.json'), 'w'), indent=1)
    print('written', os.path.join(RES, 'proxy.json'))

    def pm(c, d=3):
        return f'{c[0]:.{d}f}$\\pm${c[1]:.{d}f}'

    DSN = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
    s = ['\\begin{tabular}{l l r cc rr}', '\\toprule',
         '\\textbf{Dataset} & \\textbf{Selection signal} & \\textbf{Runs} & $\\rho$ \\textbf{vs.\\ 3-ep.} & '
         '\\textbf{Top-10 kept} & \\textbf{Hit@3} & \\textbf{AUC} \\\\',
         ' & & & & \\textbf{(\\%)} & \\textbf{(\\%)} & \\textbf{(worst 25\\%)} \\\\']
    for ds in DATASETS:
        s += ['\\midrule']
        e = out[ds]
        s.append(f"\\multirow{{3}}{{*}}{{{DSN[ds]}}} & 1-epoch accuracy (free) & {e['all']['proxy']['n_runs']} & "
                 f"{pm(e['all']['proxy']['rho'])} & {e['all']['proxy']['retention'][0]:.1f} & "
                 f"{e['all']['proxy']['hit3']:.0f} & {e['pooled']['auc_proxy']:.3f} \\\\")
        s.append(f" & LLM accuracy predictor & {e['all']['pred']['n_runs']} & "
                 f"{pm(e['all']['pred']['rho'])} & {e['all']['pred']['retention'][0]:.1f} & "
                 f"{e['all']['pred']['hit3']:.0f} & {e['pooled']['auc_pred']:.3f} \\\\")
        s.append(f" & \\textit{{random selection}} & -- & 0.000 & {e['all']['proxy']['chance']:.1f} & "
                 f"-- & 0.500 \\\\")
    s += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'tab_proxy.tex'), 'w').write('\n'.join(s))
    print('wrote tab_proxy.tex')
