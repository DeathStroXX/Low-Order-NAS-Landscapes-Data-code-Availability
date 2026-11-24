"""Two supplementary quantities that no other script computes.

1. Rank fidelity of Bench.predictor_emulator, the resampled stand-in for the LLM predictor.

The emulator is applied to the measured 3-epoch accuracy of every released architecture that
carries a predictor value (seed 0, 30 bins) and its Spearman correlation with that accuracy is
reported per dataset, next to the measured predictor's five-fold value from s16_surrogate.py.
The supplement quotes both.

2. The raw-log high-prediction contrast on CIFAR-100: the mean measured 3-epoch accuracy of
the architectures the LLM predictor scored at 90% or more, against the mean over every
architecture the 17 CIFAR-100 runs logged (common.runs, all arms).

3. What returning the architecture a measured epoch ranks first, rather than the one the LLM
predictor ranks first, is worth: per run, the 3-epoch accuracy of the 1-epoch argmax minus that
of the predictor argmax, averaged over the 17 runs of each dataset.

Writes results/extras.json.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import common, fractalbench


def main():
    sur = json.load(open(os.path.join(common.RES, 'surrogate.json')))
    out = {}
    for ds in common.DATASETS:
        b = fractalbench.Bench(ds)
        f = b.predictor_emulator()
        a3 = np.array([r['acc3'] for r in b.rows if r.get('pred') is not None])
        out[ds] = dict(n=int(len(a3)), rho_emulator=float(spearmanr(f(a3), a3).statistic),
                       rho_measured=sur[ds]['rho']['predictor_llm'])
        print(ds, out[ds])
    a3, pr = [], []
    for m in common.METHODS:
        for r in common.runs(m, 'cifar100'):
            a3 += list(r['acc3']); pr += list(r['pr'])
    a3, pr = np.array(a3), np.array(pr)
    hi = pr >= 90
    raw = dict(n=int(len(a3)), n_high=int(hi.sum()), mean_high=float(a3[hi].mean()),
               mean_all=float(a3.mean()), gap=float(a3[hi].mean() - a3.mean()))
    print('raw high band cifar100', raw)
    top1 = {}
    for ds in common.DATASETS:
        d = [r['acc3'][np.argmax(r['acc1'])] - r['acc3'][np.argmax(r['pr'])]
             for m in common.METHODS for r in common.runs(m, ds)]
        top1[ds] = dict(n_runs=len(d), mean=float(np.mean(d)))
    print('epoch-top1 over predictor-top1', top1)
    json.dump({'emulator': out, 'raw_high_band_cifar100': raw, 'top1_epoch_vs_predictor': top1},
              open(os.path.join(common.RES, 'extras.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
