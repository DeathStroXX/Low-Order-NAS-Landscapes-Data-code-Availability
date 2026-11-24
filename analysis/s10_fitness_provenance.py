#!/usr/bin/env python3
"""s10_fitness_provenance.py -- establishes, from the pod logs, which signal the GA actually
selected on in every arm.

Each evaluation writes a `>>> FITNESS SCORE: <v>% (source: <s>, checksum: <c>)` line. This script
matches every such value, by checksum, against the three per-architecture logs of the same run and
reports how often the selected fitness equals the LLM predictor's output, the 1-epoch accuracy, or
the 3-epoch accuracy. It also counts the duplicate-penalty events the pipeline applies.

This is a provenance check: the result is used in the Method section to describe the fitness signal
correctly, and in the proxy section as the motivation for replacing it.

Writes results/fitness.json and tables/tab_fitness.tex.
"""
import json, os, re, glob, collections
import numpy as np
from common import raw_runs, _pod_text, DATASETS, METHODS, RES, HERE

T = os.path.join(HERE, '..', 'tables')
PAT = re.compile(r'FITNESS SCORE:\s*([0-9.]+)%\s*\(source:\s*([^,]+),\s*checksum:\s*([0-9a-f]+)')
DUP = re.compile(r'applying (\d+)% fitness penalty')
TOL = 0.011


# Pod-log reading goes through common._pod_text, which skips the one multi-run log that was
# archived as an identical copy inside four run directories; counting it once per directory
# would multiply every event it records.


if __name__ == '__main__':
    out = {'arms': {}, 'pooled': collections.Counter(), 'duplicate_penalties': {}}
    agg = collections.defaultdict(lambda: collections.Counter())
    for m in METHODS:
        for ds in DATASETS:
            rows = []; dup = 0
            for r in raw_runs(m, ds):
                d1 = {x['uid']: x['accuracy'] for x in r['e1']}
                d3 = {x['uid']: x['accuracy'] for x in r['e3']}
                dp = {x['uid']: x['accuracy'] for x in r['pr']}
                txt = _pod_text(r['path'])
                dup += len(DUP.findall(txt))
                for mt in PAT.finditer(txt):
                    val, src, ck = float(mt.group(1)), mt.group(2).strip(), mt.group(3)
                    cand = [u for u in d1 if u.startswith(ck)] if len(ck) < 32 else ([ck] if ck in d1 else [])
                    if len(cand) != 1:
                        continue
                    u = cand[0]
                    rows.append((src, val, d1.get(u), d3.get(u), dp.get(u)))
            if not rows:
                continue
            v = np.array([x[1] for x in rows])
            def share(i):
                x = np.array([np.nan if r[i] is None else r[i] for r in rows], float)
                ok = ~np.isnan(x)
                return float(100 * np.mean(np.abs(v[ok] - x[ok]) < TOL)) if ok.sum() else float('nan')
            srcs = collections.Counter(x[0].split('(')[0].strip() for x in rows)
            out['arms'][f'{m}|{ds}'] = dict(n=len(rows), matches_predictor=share(4),
                                            matches_1epoch=share(2), matches_3epoch=share(3),
                                            mean_fitness=float(v.mean()),
                                            mean_1epoch=float(np.nanmean([x[2] for x in rows])),
                                            sources=dict(srcs))
            out['duplicate_penalties'][f'{m}|{ds}'] = dup
            for k, c in srcs.items():
                agg[m][k] += c
    out['pooled'] = {m: dict(agg[m]) for m in agg}
    out['total_duplicate_penalties'] = int(sum(out['duplicate_penalties'].values()))
    out['total_matched'] = int(sum(n for arm in out['pooled'].values() for n in arm.values()))
    json.dump(out, open(os.path.join(RES, 'fitness.json'), 'w'), indent=1)
    print('written', os.path.join(RES, 'fitness.json'))

    DSN = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
    SH = {'baseline': 'Baseline GA', 'frozen': 'Restarts, frozen ops.', 'qwen': 'Qwen2.5-Coder-7B',
          'deepseek': 'DeepSeek-Coder-6.7B', 'mistral': 'Mistral-7B'}
    s = ['\\begin{tabular}{l l r rrr}', '\\toprule',
         ' & & & \\multicolumn{3}{c}{\\textbf{Selected fitness equals (\\%)}} \\\\', '\\cmidrule(lr){4-6}',
         '\\textbf{Configuration} & \\textbf{Dataset} & \\textbf{Matched} & \\textbf{Predictor} & \\textbf{1-epoch} & \\textbf{3-epoch} \\\\', '\\midrule']
    for m in METHODS:
        for ds in DATASETS:
            e = out['arms'].get(f'{m}|{ds}')
            if not e:
                continue
            s.append(f"{SH[m]} & {DSN[ds]} & {e['n']} & {e['matches_predictor']:.1f} & "
                     f"{e['matches_1epoch']:.1f} & {e['matches_3epoch']:.1f} \\\\")
    s += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(T, 'tab_fitness.tex'), 'w').write('\n'.join(s))
    print('wrote tab_fitness.tex')
