"""Do the regret conclusions depend on the shortlist size? k = 5, 10, 20 and 50.

The paper decides its regret comparisons at a shortlist of ten. This script repeats the two
comparisons whose conclusion is stated in terms of regret at other shortlist sizes:

1. The exact degree-d ladder (results/ceiling.json, written by s59_ceiling.py): in how many of
   the eighteen ordered pairs of orders does the lower R^2 ceiling give up less accuracy, and
   what the degree-2 projection gives up on raw and on recalibrated accuracy.
2. Recalibrate or transfer (Table II): the same ridge, the same fit sets and the same seeds as
   s53_spectrum.py, with regret taken at each k; in how many of the twelve budget-by-dataset
   cells the target's own recalibrated labels give up less than the raw-scale transferred prior,
   and than the recalibrated transfer.

Writes results/k_sensitivity.json.
"""
import itertools, json, os

import numpy as np
from scipy.stats import rankdata

import common
import s53_spectrum as s53

KS = (5, 10, 20, 50)


def ladder_pairs():
    C = json.load(open(os.path.join(common.RES, 'ceiling.json')))
    out = {}
    for k in KS:
        n, g2raw, g2rec = 0, {}, {}
        for ds, v in C['datasets'].items():
            rows = v['raw']['ladder']
            for a, b in itertools.permutations(rows, 2):
                if a['ceiling_r2_percent'] < b['ceiling_r2_percent'] and \
                        a['regret'][str(k)] < b['regret'][str(k)]:
                    n += 1
            g2raw[ds] = v['raw']['ladder'][1]['regret'][str(k)]
            g2rec[ds] = v['recalibrated']['ladder'][1]['regret'][str(k)]
        out[str(k)] = {'pairs_lower_ceiling_lower_regret': n, 'g2_regret_raw': g2raw,
                       'g2_regret_recal': g2rec}
    return out


def transfer_comparison():
    G, acc = s53.load()
    X2 = s53.design(G, 2)
    hardest = {'cifar10': 'ImageNet16-120', 'cifar100': 'ImageNet16-120',
               'ImageNet16-120': 'cifar100'}

    def regret(pred, truth, k):
        return float(truth.max() - truth[np.argsort(-pred, kind='stable')[:k]].max())

    res = {str(k): {'own_beats_transfer_raw': 0, 'own_beats_transfer_recal': 0, 'cells': 0}
           for k in KS}
    detail = {}
    for tgt in s53.DATASETS:
        src = hardest[tgt]
        for n in (100, 250, 500, 1000):
            reg = {a: {k: [] for k in KS} for a in ('own', 'transfer_raw', 'transfer_recal')}
            for s_ in range(s53.SEEDS):
                rng = np.random.default_rng(5000 + s_)
                idx = rng.choice(len(G), n, replace=False)
                ev = np.setdiff1d(np.arange(len(G)), idx)
                t = acc[tgt][ev]
                preds = {
                    'own': X2 @ s53.ridge_fit(X2[idx], rankdata(acc[tgt][idx]).astype(float)),
                    'transfer_raw': X2 @ s53.ridge_fit(X2[idx], acc[src][idx]),
                    'transfer_recal': X2 @ s53.ridge_fit(X2[idx], rankdata(acc[src][idx]).astype(float)),
                }
                for a, p in preds.items():
                    for k in KS:
                        reg[a][k].append(regret(p[ev], t, k))
            for k in KS:
                m = {a: float(np.mean(reg[a][k])) for a in reg}
                r = res[str(k)]
                r['cells'] += 1
                r['own_beats_transfer_raw'] += m['own'] < m['transfer_raw']
                r['own_beats_transfer_recal'] += m['own'] < m['transfer_recal']
                detail.setdefault(tgt, {}).setdefault(str(n), {})[str(k)] = m
    return res, detail


def main():
    lad = ladder_pairs()
    tr, detail = transfer_comparison()
    out = {'ks': list(KS), 'ladder': lad, 'transfer': tr, 'transfer_detail': detail}
    json.dump(out, open(os.path.join(common.RES, 'k_sensitivity.json'), 'w'), indent=1)
    for k in KS:
        l, t = lad[str(k)], tr[str(k)]
        print('k=%-3d lower-ceiling/lower-regret pairs %2d of 18 | g2 regret raw %s recal %s | '
              'own recal beats raw transfer in %d/%d cells, recal transfer in %d/%d'
              % (k, l['pairs_lower_ceiling_lower_regret'],
                 ' '.join('%.2f' % v for v in l['g2_regret_raw'].values()),
                 ' '.join('%.2f' % v for v in l['g2_regret_recal'].values()),
                 t['own_beats_transfer_raw'], t['cells'], t['own_beats_transfer_recal'], t['cells']))


if __name__ == '__main__':
    main()
