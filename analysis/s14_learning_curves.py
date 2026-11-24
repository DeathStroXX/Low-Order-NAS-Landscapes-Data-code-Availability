#!/usr/bin/env python3
"""s14_learning_curves.py -- recovers the per-epoch learning curve of every evaluated
architecture from the pipeline's pod logs, releases it as a data file, and measures how rank
fidelity grows with the training budget.

The released JSONL records hold only the 1-epoch and the 3-epoch test accuracy. The pod logs, in
contrast, print a `Train Acc: a, Test Acc: b` pair after each of the three epochs, immediately
after the `Evaluating unique arch (checksum: ...)` line that names the architecture. Parsing the
two streams together recovers the full three-point train and test curve per architecture, which
makes it possible to ask what a second epoch, or a trivial extrapolation from the first two, buys
over a single epoch.

Outputs
  data/learning_curves.csv          one row per architecture and epoch (released for reuse)
  results/curves.json               all fidelity statistics
  tables/tab_fidelity_curve.tex     the cost/fidelity table of the main paper

Fidelity is measured against the 3-epoch test accuracy, per run, then averaged over runs, exactly
as in s11_proxy_fidelity.py. The 3-epoch *train* accuracy is deliberately excluded from the
comparison: it is measured in the same epoch as the reference, so its near-perfect correlation is
an artifact of simultaneity rather than a usable early-stopping signal.
"""
import csv, json, os, re, sys
import numpy as np
from scipy import stats
from common import raw_runs, _pod_text, DATASETS, METHODS, LLMS, RES, HERE, DATA
from statsutil import ci95

T = os.path.join(HERE, '..', 'tables')
ARCH = re.compile(r'Evaluating unique arch \(checksum:\s*([0-9a-f]+)')
EP = re.compile(r'Train Acc:\s*([0-9.]+),\s*Test Acc:\s*([0-9.]+)')
N_EPOCHS = 3
MIN_PER_RUN = 20          # a run needs this many recovered curves to enter the statistics
ARM = {'baseline': 'baseline', 'frozen': 'frozen'}


def curves(method, ds):
    """Per run: {uid: [(train, test)] * 3}, plus coverage counts."""
    out = []
    for r in raw_runs(method, ds):
        uids = {x['uid'] for x in r['e1']}
        by_prefix = {}
        for u in uids:
            by_prefix.setdefault(u[:8], []).append(u)
        cur, buf, got = None, [], {}

        def close():
            if cur is None:
                return
            cand = by_prefix.get(cur, [])
            if len(cand) == 1 and len(buf) == N_EPOCHS:
                got.setdefault(cand[0], list(buf))

        for line in _pod_text(r['path']).split('\n'):
            a = ARCH.search(line)
            if a:
                close(); cur, buf = a.group(1), []
                continue
            e = EP.search(line)
            if e and cur is not None:
                buf.append((float(e.group(1)) * 100.0, float(e.group(2)) * 100.0))
        close()
        out.append(dict(dir=r['dir'], curves=got, n_logged=len(uids), n_curves=len(got)))
    return out


def signals(cv, d3):
    """The candidate selection signals, all derivable from the first one or two epochs."""
    u = [k for k in cv if k in d3]
    e1 = np.array([cv[k][0][1] for k in u])
    e2 = np.array([cv[k][1][1] for k in u])
    return u, {
        'e1': e1,                      # one epoch of training
        'e2': e2,                      # two epochs
        'lin': e2 + (e2 - e1),         # two epochs plus a 2-point linear extrapolation
    }


def fidelity(sig, y, k=10):
    rho = stats.spearmanr(sig, y).statistic
    top_t, top_s = set(np.argsort(y)[-k:]), set(np.argsort(sig)[-k:])
    rank = np.argsort(np.argsort(-y))
    b = int(np.argmax(sig))
    return rho, 100 * len(top_t & top_s) / k, bool(rank[b] < 3), 100 * k / len(y)


LABEL = {'e1': '1 epoch', 'e2': '2 epochs', 'lin': '2 epochs $+$ extrapolation',
         'pred': 'LLM accuracy predictor'}
ORDER = ['e1', 'e2', 'lin', 'pred']

if __name__ == '__main__':
    rows = []
    out = {'coverage': {}, 'fidelity': {}}
    for ds in DATASETS:
        per = {s: [] for s in ORDER}
        full = {s: [] for s in ORDER}      # runs whose curve coverage is complete
        cov_logged = cov_curves = 0
        n_runs_used = n_runs_total = 0
        partial = []
        for m in METHODS:
            arm = ARM.get(m, 'llm')
            for cv_run in curves(m, ds):
                cov_logged += cv_run['n_logged']; cov_curves += cv_run['n_curves']
                n_runs_total += 1
                rr = [r for r in raw_runs(m, ds) if r['dir'] == cv_run['dir']][0]
                d3 = {x['uid']: x['accuracy'] for x in rr['e3']}
                dp = {x['uid']: x['accuracy'] for x in rr['pr']}
                cv = cv_run['curves']
                for u, c in cv.items():
                    for i, (tr, te) in enumerate(c, start=1):
                        rows.append(dict(uid=u, dataset=ds, method=m, arm=arm,
                                         run=f'{m}/{ds}/{cv_run["dir"]}', epoch=i,
                                         train_acc=round(tr, 4), test_acc=round(te, 4),
                                         acc_epoch3_true=d3.get(u), predicted_fitness=dp.get(u)))
                u, sg = signals(cv, d3)
                if len(u) < MIN_PER_RUN:
                    continue
                n_runs_used += 1
                complete = cv_run['n_curves'] == cv_run['n_logged']
                if not complete:
                    partial.append(dict(run=f'{m}/{cv_run["dir"]}', curves=cv_run['n_curves'],
                                        logged=cv_run['n_logged']))
                y = np.array([d3[k] for k in u])
                sg['pred'] = np.array([dp[k] for k in u])
                for s in ORDER:
                    f = fidelity(sg[s], y)
                    per[s].append(f)
                    if complete:
                        full[s].append(f)
        out['coverage'][ds] = dict(logged=cov_logged, curves=cov_curves,
                                   pct=100 * cov_curves / cov_logged,
                                   runs_used=n_runs_used, runs_total=n_runs_total,
                                   partial=partial)
        if not per[ORDER[0]]:
            sys.stderr.write(
                '\ns14_learning_curves: no learning curve could be recovered for %s.\n'
                'This script reads the per-epoch progress lines of the pod logs, and the copy\n'
                'of logs/ in this archive has those lines collapsed to keep the download to one\n'
                'file (see logs/ABOUT_THESE_LOGS.txt). It is the one script whose input the\n'
                'compaction removes. The curves it recovered are released as\n'
                'data/learning_curves.csv, and every downstream number -- the fidelity table,\n'
                'the halving frontier and the warm start -- is computed from that file by\n'
                's17_halving.py and s29_warmstart.py, which run from this archive unchanged.\n'
                'Rerun this script only against an uncompacted copy of the logs.\n' % ds)
            sys.exit(2)
        out['fidelity'][ds] = {s: dict(n_runs=len(per[s]),
                                       rho=ci95([a[0] for a in per[s]]),
                                       rho_min=float(min(a[0] for a in per[s])),
                                       retention=ci95([a[1] for a in per[s]]),
                                       hit3=100 * float(np.mean([a[2] for a in per[s]])),
                                       chance=float(np.mean([a[3] for a in per[s]])))
                               for s in ORDER}
        # Sensitivity: the same statistics over only the runs with complete curve coverage.
        out.setdefault('fidelity_complete_only', {})[ds] = {
            s: dict(n_runs=len(full[s]), rho=ci95([a[0] for a in full[s]]),
                    retention=ci95([a[1] for a in full[s]]),
                    hit3=100 * float(np.mean([a[2] for a in full[s]])))
            for s in ORDER if len(full[s]) > 1}
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, 'learning_curves.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['uid', 'dataset', 'method', 'arm', 'run', 'epoch',
                                           'train_acc', 'test_acc', 'acc_epoch3_true',
                                           'predicted_fitness'], lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows, {len(rows)//N_EPOCHS} architectures)')
    json.dump(out, open(os.path.join(RES, 'curves.json'), 'w'), indent=1)
    print('written', os.path.join(RES, 'curves.json'))

    def pm(c, d=3):
        return f'{c[0]:.{d}f}$\\pm${c[1]:.{d}f}'

    DSN = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
    s = ['\\begin{tabular}{l l r cc r}', '\\toprule',
         '\\textbf{Dataset} & \\textbf{Selection signal} & \\textbf{Cost} & $\\rho$ \\textbf{vs.\\ 3-ep.} & '
         '\\textbf{Top-10 kept} & \\textbf{Hit@3} \\\\',
         ' & & \\textbf{(epochs)} & & \\textbf{(\\%)} & \\textbf{(\\%)} \\\\']
    COST = {'e1': '1', 'e2': '2', 'lin': '2', 'mean12': '2', 'pred': '3 $+$ LLM'}
    for ds in DATASETS:
        s += ['\\midrule', f'\\multirow{{{len(ORDER) + 1}}}{{*}}{{{DSN[ds]}}}']
        first = True
        for sig in ORDER:
            e = out['fidelity'][ds][sig]
            lead = '' if first else ''
            s[-1] = s[-1] if first else s[-1]
            s.append(f"{lead} & {LABEL[sig]} & {COST[sig]} & {pm(e['rho'])} & "
                     f"{e['retention'][0]:.1f} & {e['hit3']:.0f} \\\\")
            first = False
        e = out['fidelity'][ds]['e1']
        s.append(f" & \\textit{{random selection}} & 0 & 0.000 & {e['chance']:.1f} & -- \\\\")
    s += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'tab_fidelity_curve.tex'), 'w').write('\n'.join(s))
    print('wrote tab_fidelity_curve.tex')
    for ds in DATASETS:
        c = out['coverage'][ds]
        print(f"  {ds}: curves for {c['curves']}/{c['logged']} architectures ({c['pct']:.1f}%), "
              f"{c['runs_used']}/{c['runs_total']} runs usable")
        for sig in ORDER:
            e = out['fidelity'][ds][sig]
            print(f"     {LABEL[sig]:30s} rho={e['rho'][0]:.3f}+-{e['rho'][1]:.3f} "
                  f"min={e['rho_min']:.3f} ret={e['retention'][0]:.1f}% hit3={e['hit3']:.0f}%")
