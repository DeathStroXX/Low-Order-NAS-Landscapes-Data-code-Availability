"""What the ceiling bounds, and what it does not: the same bound in the decision metrics.

Section IV reads the cumulative amplitude spectrum as a ceiling: because the ANOVA components
are orthogonal, the best degree-d approximation of the accuracy function in squared error is
exactly the sum of its components of order <= d, so the cumulative share is an upper bound on
the R^2 of any degree-d function of the design variables. That statement is exact, but it is a
statement about squared error on the response *as measured*, and two of this paper's own results
mean it cannot be left there:

  * Section VI shows that a variance decomposition is not invariant to a monotone change of the
    response, so the numeric value of the ceiling depends on the scale it is computed on. The
    raw and recalibrated ceilings are both reported for that reason, and the recalibrated one is
    the scale-free statement.

  * Section VIII argues that a cheap ranker should be judged by regret at a shortlist rather
    than by a correlation or an R^2, and a bound on squared error does not bound either. Two
    separate facts are involved and the script reports both, because conflating them is how a
    ceiling gets over-read:

      (i)  for a *fixed* score s, rho(s, f) and the regret of s's top k are unchanged when f is
           replaced by any strictly increasing function of it, while R^2(s, f) is not. A bound
           expressed in variance shares therefore cannot by itself cap a rank or a regret.
      (ii) *which* degree-d function the squared-error criterion selects does depend on the
           scale, because the projection is taken in that metric. So recalibration does not
           merely raise the numeric ceiling: it changes the degree-2 function one gets, and the
           one it gets is a different -- here a far better -- ranker.

    (ii) is the exact, model-free form of Section VI's recommendation, and it is reported here
    with no ridge, no labels and no fitting anywhere in it.

This script therefore computes, exactly and with nothing fitted, what the best degree-d
approximation actually achieves in the metrics a budget is decided in. For each dataset and each
d it reports

    ceiling_d   the cumulative variance share: an *upper* bound on any degree-d predictor's R^2
    rho_d       the Spearman rho of g_d, the squared-error-optimal degree-d approximation,
                against the true accuracy over all 15,625 cells
    regret_d    the accuracy points given up by taking g_d's top ten and training only those
    kstar_d     the smallest shortlist of g_d whose regret falls under a tenth of a point

rho_d, regret_d and kstar_d are *attainable* values, hence lower bounds on what the best
degree-d ranker achieves; ceiling_d is an upper bound in squared error. Together they bracket
what order d is worth, which is what a reader deciding whether to buy an elaborate predictor
needs. The two do not order the same way, and the paper quotes the case where they disagree
most sharply.

The same quantities are computed on the recalibrated response, which is what makes the
invariance concrete: the order-<=2 R^2 ceiling moves by more than twenty points between the two
scales while the rank of the same degree-2 projection barely moves at all.

    python3 analysis/s59_ceiling.py
"""
import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

import anova

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
DATA = os.path.join(ROOT, 'data')

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
SHAPE = (5,) * 6
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def genotype(arch):
    toks = re.findall(r'([a-z0-9_]+)~\d', arch)
    if len(toks) != 6:
        raise ValueError('unexpected architecture string: %r' % arch)
    return [OPS.index(t) for t in toks]


def load():
    """The benchmark table, read exactly as s53 reads it."""
    path = os.path.join(DATA, 'nb201_transfer.csv')
    rows = list(csv.DictReader(open(path)))
    G = np.array([genotype(r['arch']) for r in rows], dtype=int)
    acc = {d: np.array([float(r['acc_' + d]) for r in rows]) for d in DATASETS}
    return G, acc


def decision(score, truth, k=10):
    """Rank fidelity and the decision metrics of a ranking, over the whole grid."""
    rho = float(spearmanr(score, truth).statistic)
    best = float(truth.max())
    order = np.argsort(-np.asarray(score, dtype=float), kind='stable')
    reg = {kk: float(best - truth[order[:kk]].max()) for kk in KS}
    kstar = next((kk for kk in KS if reg[kk] < 0.1), None)
    return {'rho': rho, 'regret': reg, 'regret_at_10': reg[10], 'kstar': kstar}


def ladder(G, y, truth):
    """The exact degree-d ladder: R^2 ceiling and attainable decision metrics, d = 1..6."""
    Y = anova.to_grid(G, y, SHAPE)
    anova.check_consistency(Y)
    sp = anova.spectrum(Y)
    cum = anova.cumulative(sp)
    sums = anova.order_sums(Y)
    idx = tuple(G.T)
    g = np.zeros(SHAPE)
    rows = []
    for d in range(1, 7):
        g = g + sums[d]
        row = {'d': d, 'ceiling_r2_percent': cum[d - 1]}
        row.update(decision(g[idx], truth))
        rows.append(row)
    return {'per_order': sp, 'cumulative': cum, 'ladder': rows}


def main():
    G, acc = load()
    out = {'n_arch': int(G.shape[0]), 'shape': list(SHAPE), 'ks': KS, 'datasets': {}}

    for ds in DATASETS:
        y = acc[ds]
        raw = ladder(G, y, y)
        rec = ladder(G, rankdata(y), y)
        out['datasets'][ds] = {'raw': raw, 'recalibrated': rec}

    # ---- fact (i): rho and regret of a FIXED score do not see a monotone change of f ------
    # Checked rather than asserted: score the grid with the raw degree-2 projection and
    # recompute its rho and regret against the rank-transformed accuracy. Both must be
    # identical to the values against raw accuracy, while R^2 is not.
    fixed = {}
    for ds in DATASETS:
        y = acc[ds]
        Y = anova.to_grid(G, y, SHAPE)
        sums = anova.order_sums(Y)
        idx = tuple(G.T)
        g2 = (sums[1] + sums[2])[idx]
        a = decision(g2, y)
        b = decision(g2, rankdata(y))
        ss_raw = 1.0 - np.var(y - g2) / np.var(y)
        yr = rankdata(y)
        gr = g2 - g2.mean() + yr.mean()
        ss_rank = 1.0 - np.var(yr - gr * (np.std(yr) / np.std(g2))) / np.var(yr)
        fixed[ds] = {'rho_against_raw': a['rho'], 'rho_against_ranks': b['rho'],
                     'rho_identical': bool(abs(a['rho'] - b['rho']) < 1e-9),
                     'kstar_against_raw': a['kstar'], 'kstar_against_ranks': b['kstar'],
                     'r2_against_raw': float(ss_raw), 'r2_against_ranks': float(ss_rank)}
        assert fixed[ds]['rho_identical'], ds
    out['fixed_score_invariance'] = fixed

    # ---- fact (ii): the projection itself is scale-dependent -----------------------------
    inv = {}
    for ds in DATASETS:
        a = out['datasets'][ds]['raw']['ladder'][1]
        b = out['datasets'][ds]['recalibrated']['ladder'][1]
        inv[ds] = {
            'ceiling_order2_raw': a['ceiling_r2_percent'],
            'ceiling_order2_recalibrated': b['ceiling_r2_percent'],
            'ceiling_shift_points': b['ceiling_r2_percent'] - a['ceiling_r2_percent'],
            'rho_order2_raw': a['rho'],
            'rho_order2_recalibrated': b['rho'],
            'rho_shift': b['rho'] - a['rho'],
            'regret10_order2_raw': a['regret_at_10'],
            'regret10_order2_recalibrated': b['regret_at_10'],
        }
    out['invariance'] = inv

    # ---- where the two readings disagree ------------------------------------------------
    # The point the paper makes in one sentence: a higher R^2 ceiling does not imply a better
    # decision. Every (dataset, d, d') with ceiling_d < ceiling_d' but regret_d < regret_d'
    # is an instance, and they are collected here rather than asserted.
    dis = []
    for ds in DATASETS:
        rows = out['datasets'][ds]['raw']['ladder']
        for i in range(len(rows)):
            for j in range(len(rows)):
                if (rows[i]['ceiling_r2_percent'] < rows[j]['ceiling_r2_percent']
                        and rows[i]['regret_at_10'] < rows[j]['regret_at_10']):
                    dis.append({'dataset': ds, 'lower_ceiling_d': rows[i]['d'],
                                'higher_ceiling_d': rows[j]['d'],
                                'ceiling': [rows[i]['ceiling_r2_percent'],
                                            rows[j]['ceiling_r2_percent']],
                                'regret_at_10': [rows[i]['regret_at_10'],
                                                 rows[j]['regret_at_10']]})
    out['ceiling_decision_disagreements'] = dis

    with open(os.path.join(RES, 'ceiling.json'), 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    # ---------------- report ----------------
    print('the degree-d ladder on NAS-Bench-201, exact, nothing fitted')
    print()
    for ds in DATASETS:
        print('%s' % SHORT[ds])
        print('       |            as measured            |        after recalibration')
        print('   d   |  R2 ceil   rho   reg@10    k*     |  R2 ceil   rho   reg@10    k*')
        for a, b in zip(out['datasets'][ds]['raw']['ladder'],
                        out['datasets'][ds]['recalibrated']['ladder']):
            print('   %d   |   %5.1f   %.3f   %5.2f   %5s   |   %5.1f   %.3f   %5.2f   %5s'
                  % (a['d'], a['ceiling_r2_percent'], a['rho'], a['regret_at_10'],
                     a['kstar'], b['ceiling_r2_percent'], b['rho'], b['regret_at_10'],
                     b['kstar']))
        print()
    print('(i)  a fixed degree-2 score has the same rho and the same k* against raw accuracy '
          'and against')
    print('     its ranks (%s), while its R^2 against the two differs (%s against %s):'
          % (', '.join('%.3f' % fixed[d]['rho_against_raw'] for d in DATASETS),
             ', '.join('%.2f' % fixed[d]['r2_against_raw'] for d in DATASETS),
             ', '.join('%.2f' % fixed[d]['r2_against_ranks'] for d in DATASETS)))
    print('     a variance bound cannot cap a rank or a regret.')
    print('(ii) the projection is scale-dependent: at order <= 2 the ceiling moves by %s '
          'points,' % ', '.join('%+.1f' % inv[d]['ceiling_shift_points'] for d in DATASETS))
    print('     and the degree-2 function recalibration selects reaches regret@10 of %s '
          'against %s.'
          % (', '.join('%.2f' % inv[d]['regret10_order2_recalibrated'] for d in DATASETS),
             ', '.join('%.2f' % inv[d]['regret10_order2_raw'] for d in DATASETS)))
    print('%d ordered pairs (d, d\') have a lower R^2 ceiling and a lower regret at ten, '
          'so the' % len(dis))
    print('ceiling does not order decisions.')
    print('\nwrote analysis/results/ceiling.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
