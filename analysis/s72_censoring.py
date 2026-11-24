"""Left truncation of the per-architecture record, per configuration (arm x dataset).

Every architecture the pipeline begins to train is announced in its run's pod log by an
`Evaluating unique arch (checksum: ...)` line. One whose first epoch ends below the dataset's
accuracy floor is dropped before anything about it reaches the JSONL logs. The censored share
of a configuration is therefore (attempted - recorded) / attempted, with `attempted` the
distinct checksums its pod logs announce and `recorded` the distinct architectures its JSONL
logs hold. Counting the pod logs' own "Accuracy is too low" error lines instead gives the
same rates to within 0.4 points.

Reads logs/ and large_input_data/pod_logs/ (all 34 runs of the October 2026 release).
Writes results/censoring.json and tables/tab_censoring.tex (inlined in the supplement).
"""
import json, os, re

import common

ARCH = re.compile(r'Evaluating unique arch \(checksum:\s*([0-9a-f]+)')
NAME = {'baseline': 'Baseline GA', 'frozen': 'Restarts, frozen operators',
        'qwen': 'Restarts, Qwen operators', 'deepseek': 'Restarts, DeepSeek operators',
        'mistral': 'Restarts, Mistral operators'}


def main():
    out = {'by_config': {}, 'range': {}}
    for ds in common.DATASETS:
        rates = []
        for m in common.METHODS:
            runs = common.raw_runs(m, ds)
            att = rec = 0
            for r in runs:
                att += len(set(ARCH.findall(common._pod_text(r['path']))))
                rec += len({x['uid'] for x in r['e1']})
            rate = (att - rec) / att
            rates.append(rate)
            out['by_config'][f'{m}/{ds}'] = dict(runs=len(runs), attempted=att, recorded=rec,
                                                  attempted_per_run=att / len(runs),
                                                  recorded_per_run=rec / len(runs), rate=rate)
        out['range'][ds] = [min(rates), max(rates)]
    json.dump(out, open(os.path.join(common.RES, 'censoring.json'), 'w'), indent=1)

    L = [r'\begin{center}\footnotesize', r'\begin{tabular}{l rr r}', r'\toprule',
         r'\textbf{Configuration} & \textbf{Attempted/run} & \textbf{Recorded/run} & '
         r'\textbf{Censored} \\', r'\midrule']
    for k, ds in enumerate(common.DATASETS):
        L += [r'\multicolumn{4}{c}{\textit{%s}} \\' % {'cifar10': 'CIFAR-10',
                                                         'cifar100': 'CIFAR-100'}[ds], r'\midrule']
        for m in common.METHODS:
            c = out['by_config'][f'{m}/{ds}']
            L.append(r'%s & %.1f & %.1f & %.1f\%% \\' % (NAME[m], c['attempted_per_run'],
                                                         c['recorded_per_run'], 100 * c['rate']))
        if k == 0:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_censoring.tex'), 'w').write('\n'.join(L) + '\n')
    for ds in common.DATASETS:
        lo, hi = out['range'][ds]
        print(f'{ds}: censored {100 * lo:.1f}--{100 * hi:.1f}% of attempts')
    print('wrote results/censoring.json and tables/tab_censoring.tex')


if __name__ == '__main__':
    main()
