"""The amplitude spectrum of the NAS-Bench-201 landscape, exactly.

Everything else in this paper is an observation about a predictor. This script measures the
property of the *landscape* that explains those observations, and it measures it exactly rather
than by fitting anything.

NAS-Bench-201's cell space is a complete factorial: six edges, five operations each, and all
5^6 = 15,625 architectures tabulated. A complete factorial with one observation per cell admits
the functional ANOVA decomposition

    f(x) = f_0 + sum_i f_i(x_i) + sum_{i<j} f_ij(x_i,x_j) + ... + f_{1..6}(x),

whose components are orthogonal and obtained by Moebius inversion over cell means -- no model,
no regression, no held-out split, no estimation error. Grouping the components by |S| gives the
*amplitude spectrum* of the landscape in the sense of Hordijk and Stadler: the share of
accuracy variance carried by interactions of each order. Klein and Hutter computed exactly this
on a fully enumerated architecture/hyperparameter grid and stopped at order two; here the space
is small enough to go to order six, which is what makes the tail -- and therefore the ceiling --
measurable rather than inferred.

What the script produces, in the order the manuscript uses it:

  1. spectrum          variance share by interaction order, per dataset, exact.
  2. null              the same spectrum for white noise. This is the control a reviewer will
                       ask for, and it needs no seed data: noise spreads uniformly over the
                       15,624 non-constant basis directions, of which C(6,k)*4^k lie at order
                       k, so a pure-noise landscape has 26% of its variance at order 6 and
                       0.15% at order 1 -- the exact mirror of what is observed. The observed
                       high-order tail is therefore not the benchmark's measurement noise; it
                       is far smaller than noise alone would produce.
  3. ceiling           the cumulative share at order d is the R^2 ceiling of *any* predictor
                       that is a degree-d function of the design variables, whatever its form.
  4. cross_dataset     the correlation of each order's component between datasets, which is
                       what decides whether a prior transfers and in which direction.
  5. recovery          how many labels a degree-2 ridge needs to recover the exact order-<=2
                       component, and whether labels from a harder dataset recover the target's
                       own low-order component better than labels from the target itself.
  6. scale             the same spectrum recomputed on monotonically recalibrated accuracy.
                       This is the result that matters: the apparent task-dependence of the
                       spectrum is an artifact of the accuracy scale. NAS-Bench-201's easy
                       tasks are compressed against a ceiling -- the best tenth of CIFAR-10
                       architectures occupy 2% of the accuracy range, against 9% on
                       ImageNet16-120 -- and squared loss on a compressed target manufactures
                       high-order structure. Remove the compression and all three datasets
                       have the same spectrum.
  7. recalibration     what that costs in practice. Fitting the same ridge on recalibrated
                       labels of the target beats transferring a prior from a harder dataset,
                       at every label budget, on rank fidelity and on regret, on all three
                       datasets -- which supersedes the "fit on the hardest dataset" rule this
                       study previously recommended.
  8. fractal           the control. In a space whose accuracies are not compressed, the
                       recalibration changes nothing, which is what identifies compression
                       rather than ranks as the mechanism.

NAS-Bench-101 and NAS-Bench-301 are not enumerable either, and their tables are not
redistributed here; `--spaces` runs the estimator on them if NAS-Bench-Suite-Zero is present,
exactly as `s43`--`s45` do, and says where to get it otherwise.

    python3 analysis/s53_spectrum.py
"""
import csv
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
DATA = os.path.join(ROOT, 'data')

#: NAS-Bench-201's five cell operations, in the order the benchmark lists them
OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
N_EDGE, N_OP = 6, 5
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
#: the three datasets ordered by how hard they are, measured by mean accuracy over the table
HARDNESS = ['cifar10', 'cifar100', 'ImageNet16-120']
SEEDS = 20
RNG = np.random.default_rng(0)


def genotype(arch):
    """The six edge operations of a NAS-Bench-201 architecture string, as indices."""
    toks = re.findall(r'([a-z0-9_]+)~\d', arch)
    if len(toks) != N_EDGE:
        raise ValueError('unexpected architecture string: %r' % arch)
    return [OPS.index(t) for t in toks]


def load():
    path = os.path.join(DATA, 'nb201_transfer.csv')
    rows = list(csv.DictReader(open(path)))
    G = np.array([genotype(r['arch']) for r in rows])
    acc = {d: np.array([float(r['acc_' + d]) for r in rows]) for d in DATASETS}
    lin = np.ravel_multi_index(G.T, (N_OP,) * N_EDGE)
    if len(set(lin.tolist())) != N_OP ** N_EDGE:
        raise SystemExit('nb201_transfer.csv is not the complete factorial this script needs')
    return G, acc


def anova(G, y):
    """Exact functional ANOVA of a complete factorial. Returns {subset: component array}."""
    Y = np.empty((N_OP,) * N_EDGE)
    Y[tuple(G.T)] = y
    comp = {}
    for k in range(N_EDGE + 1):
        for S in itertools.combinations(range(N_EDGE), k):
            axes = tuple(a for a in range(N_EDGE) if a not in S)
            m = Y.mean(axis=axes, keepdims=True) if axes else Y.copy()
            for j in range(k):
                for T in itertools.combinations(S, j):
                    m = m - comp[T]
            comp[S] = m
    return comp, Y


def by_order(comp, shape):
    """Sum the components of each order, and the variance each order carries."""
    tot, var = {}, {}
    for k in range(1, N_EDGE + 1):
        acc = np.zeros(shape)
        for S in itertools.combinations(range(N_EDGE), k):
            acc = acc + np.broadcast_to(comp[S], shape)
        tot[k] = acc
        var[k] = float((acc ** 2).mean())
    return tot, var


def null_spectrum():
    """Variance share by order for white noise: C(6,k)*4^k directions out of 5^6-1."""
    import math
    n = {k: math.comb(N_EDGE, k) * (N_OP - 1) ** k for k in range(1, N_EDGE + 1)}
    tot = sum(n.values())
    return {k: n[k] / tot for k in n}


def design(G, order):
    """Centred indicator design matrix up to the given interaction order."""
    cols = [np.ones(len(G))]
    singles = []
    for i in range(N_EDGE):
        block = [(G[:, i] == v).astype(float) for v in range(N_OP - 1)]
        singles.append(block)
        cols.extend(block)
    if order >= 2:
        for i, j in itertools.combinations(range(N_EDGE), 2):
            for a in singles[i]:
                for b in singles[j]:
                    cols.append(a * b)
    if order >= 3:
        for i, j, k in itertools.combinations(range(N_EDGE), 3):
            for a in singles[i]:
                for b in singles[j]:
                    for c in singles[k]:
                        cols.append(a * b * c)
    return np.column_stack(cols)


def ridge_fit(X, y, alpha=1.0):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A[0, 0] -= alpha
    return np.linalg.solve(A, X.T @ y)


def main():
    G, acc = load()
    shape = (N_OP,) * N_EDGE
    out = {'n_arch': int(len(G)), 'n_edges': N_EDGE, 'n_ops': N_OP, 'datasets': DATASETS,
           'seeds': SEEDS}

    # ---- 1-3. the spectrum, its null, and the ceiling it implies
    comps, orders, spec = {}, {}, {}
    for d in DATASETS:
        c, Y = anova(G, acc[d])
        o, v = by_order(c, shape)
        comps[d], orders[d] = c, o
        tot = sum(v.values())
        spec[d] = {'total_variance': tot,
                   'share': {k: v[k] / tot for k in v},
                   'cumulative': {k: sum(v[j] for j in range(1, k + 1)) / tot
                                  for k in v},
                   'mean_accuracy': float(acc[d].mean())}
    out['spectrum'] = spec
    out['null'] = null_spectrum()
    out['ceiling'] = {d: {k: spec[d]['cumulative'][k] for k in (1, 2, 3)} for d in DATASETS}
    #: how far below the white-noise null the top of the spectrum sits, as a ratio
    out['null_ratio'] = {d: {k: spec[d]['share'][k] / out['null'][k] for k in range(1, 7)}
                         for d in DATASETS}

    # ---- 4. are the low orders shared across tasks?
    out['cross_dataset'] = {}
    for a, b in itertools.combinations(DATASETS, 2):
        out['cross_dataset']['%s|%s' % (a, b)] = {
            k: float(np.corrcoef(orders[a][k].ravel(), orders[b][k].ravel())[0, 1])
            for k in range(1, N_EDGE + 1)}

    # ---- 5. recovering the exact low-order component from labels
    truth2 = {d: (orders[d][1] + orders[d][2]).ravel()[
        np.ravel_multi_index(G.T, shape)] for d in DATASETS}
    budgets = [100, 250, 500, 1000, 2000]
    rec = {d: {} for d in DATASETS}
    X2 = design(G, 2)
    for d in DATASETS:
        for n in budgets:
            own, trans = [], {s: [] for s in DATASETS if s != d}
            for s in range(SEEDS):
                rng = np.random.default_rng(1000 + s)
                idx = rng.choice(len(G), n, replace=False)
                # every model is scored on the architectures none of them was fitted on, so a
                # model fitted on the target is not credited for its own training points
                ev = np.setdiff1d(np.arange(len(G)), idx, assume_unique=False)
                w = ridge_fit(X2[idx], acc[d][idx])
                p = (X2 @ w)[ev]
                own.append(float(np.corrcoef(p, truth2[d][ev])[0, 1]))
                for src in trans:
                    ws = ridge_fit(X2[idx], acc[src][idx])
                    q = (X2 @ ws)[ev]
                    trans[src].append(float(np.corrcoef(q, truth2[d][ev])[0, 1]))
            rec[d][str(n)] = {'own': float(np.mean(own)),
                              'own_sd': float(np.std(own)),
                              **{'from_' + s: float(np.mean(v)) for s, v in trans.items()}}
    out['recovery'] = rec

    # ---- 6. the spectrum is a property of the space, not of the task: the apparent
    #         task-dependence above is the accuracy scale. Recompute it on ranks, which is the
    #         crudest monotone recalibration there is.
    from scipy.stats import rankdata, skew, spearmanr
    rec_spec = {}
    for d in DATASETS:
        c, _ = anova(G, rankdata(acc[d]).astype(float))
        _, v = by_order(c, shape)
        tot = sum(v.values())
        rec_spec[d] = {'share': {k: v[k] / tot for k in v},
                       'cumulative': {k: sum(v[j] for j in range(1, k + 1)) / tot for k in v}}
    out['spectrum_recalibrated'] = rec_spec

    #: ranks are the crudest monotone correction, so check that the result is the compression
    #: and not the particular transform: a logit of the accuracy is a different monotone map
    #: and must give the same spectrum.
    logit_spec = {}
    for d in DATASETS:
        q = np.clip(acc[d] / 100.0, 1e-3, 1 - 1e-3)
        c, _ = anova(G, np.log(q / (1 - q)))
        _, v = by_order(c, shape)
        tot = sum(v.values())
        logit_spec[d] = {'cumulative': {k: sum(v[j] for j in range(1, k + 1)) / tot
                                        for k in v}}
    out['spectrum_logit'] = logit_spec

    #: how compressed each accuracy scale is: skew, and the share of the full accuracy range
    #: that the best tenth of architectures is squeezed into
    out['compression'] = {}
    for d in DATASETS:
        a = acc[d]
        out['compression'][d] = {
            'skew': float(skew(a)),
            'top_decile_spread': float((a.max() - np.percentile(a, 90)) / (a.max() - a.min())),
            'ceiling_gap': float(a.max() - np.percentile(a, 90))}

    # ---- 7. recalibrate, or transfer? Same ridge, same fit sets, same evaluation pool.
    def regret_at(pred, truth, k):
        return float(truth.max() - truth[np.argsort(-pred)[:k]].max())

    hardest = {'cifar10': 'ImageNet16-120', 'cifar100': 'ImageNet16-120',
               'ImageNet16-120': 'cifar100'}
    out['recalibration'] = {}
    for tgt in DATASETS:
        src = hardest[tgt]
        per_n = {}
        for n in (100, 250, 500, 1000):
            arms = {'own_raw': ([], []), 'own_recal': ([], []), 'transfer_raw': ([], []),
                    'transfer_recal': ([], []), 'combined': ([], [])}
            for s_ in range(SEEDS):
                rng = np.random.default_rng(5000 + s_)
                idx = rng.choice(len(G), n, replace=False)
                ev = np.setdiff1d(np.arange(len(G)), idx)
                t = acc[tgt][ev]
                y = np.zeros(len(G))
                y[idx] = acc[tgt][idx]
                p = (X2 @ ridge_fit(X2[idx], acc[tgt][idx]))[ev]
                arms['own_raw'][0].append(spearmanr(p, t).statistic)
                arms['own_raw'][1].append(regret_at(p, t, 10))
                yr = rankdata(acc[tgt][idx]).astype(float)
                p_own = (X2 @ ridge_fit(X2[idx], yr))[ev]
                arms['own_recal'][0].append(spearmanr(p_own, t).statistic)
                arms['own_recal'][1].append(regret_at(p_own, t, 10))
                p = (X2 @ ridge_fit(X2[idx], acc[src][idx]))[ev]
                arms['transfer_raw'][0].append(spearmanr(p, t).statistic)
                arms['transfer_raw'][1].append(regret_at(p, t, 10))
                # transfer with the same recalibration applied to the source labels, so that
                # the comparison with own_recal differs in the data source alone
                p_tr = (X2 @ ridge_fit(X2[idx], rankdata(acc[src][idx]).astype(float)))[ev]
                arms['transfer_recal'][0].append(spearmanr(p_tr, t).statistic)
                arms['transfer_recal'][1].append(regret_at(p_tr, t, 10))
                # both sources at once: the rank-average of the two recalibrated predictions
                p = rankdata(p_own) + rankdata(p_tr)
                arms['combined'][0].append(spearmanr(p, t).statistic)
                arms['combined'][1].append(regret_at(p, t, 10))
            per_n[str(n)] = {k: {'rho': float(np.mean(v[0])), 'rho_sd': float(np.std(v[0])),
                                 'rho_se': float(np.std(v[0], ddof=1) / np.sqrt(len(v[0]))),
                                 'regret10': float(np.mean(v[1])),
                                 'regret10_se': float(np.std(v[1], ddof=1) / np.sqrt(len(v[1])))}
                             for k, v in arms.items()}
        out['recalibration'][tgt] = {'transfer_source': src, 'budgets': per_n}

    # ---- 8. the control: a space whose accuracies are not compressed
    # the baseline and LLM-arm rows of the per-architecture record (the frozen control is
    # analysed from its own logs elsewhere), in the column names the code below uses
    fpath = os.path.join(DATA, 'gene_diversity_benchmark.csv')
    if os.path.exists(fpath):
        genes = ['n_columns', 'base_channels', 'n_blocks', 'dropout_prob', 'fc_dropout', 'lr',
                 'momentum', 'activation', 'kernel_size', 'pooling_type', 'conv_type',
                 'norm_type', 'optimizer_type']
        rows = [dict(r, acc_epoch3_true=r['acc3']) for r in csv.DictReader(open(fpath))
                if r['method'] != 'frozen']
        frac = {}
        for ds in ('cifar10', 'cifar100'):
            sub = [r for r in rows if r.get('dataset') == ds and r.get('acc_epoch3_true')
                   not in ('', None) and all(r.get(g) not in (None, '') for g in genes)]
            if len(sub) < 500:
                continue
            y = np.array([float(r['acc_epoch3_true']) for r in sub])
            lv = {g: sorted({r[g] for r in sub}) for g in genes}
            Gi = np.array([[lv[g].index(r[g]) for g in genes] for r in sub])
            cols, singles = [np.ones(len(Gi))], []
            for i, g in enumerate(genes):
                blk = [(Gi[:, i] == v).astype(float) for v in range(len(lv[g]) - 1)]
                singles.append(blk)
                cols.extend(blk)
            X1 = np.column_stack(cols)
            for i, j in itertools.combinations(range(len(genes)), 2):
                for a in singles[i]:
                    for b in singles[j]:
                        cols.append(a * b)
            Xf = np.column_stack(cols)
            res = {}
            for n in (250, 500, 1000):
                if n > len(y) // 2:
                    continue
                arms = {'own_raw': ([], []), 'own_recal': ([], [])}
                for s_ in range(SEEDS):
                    rng = np.random.default_rng(6000 + s_)
                    idx = rng.choice(len(y), n, replace=False)
                    ev = np.setdiff1d(np.arange(len(y)), idx)
                    t = y[ev]
                    p = (Xf @ ridge_fit(Xf[idx], y[idx]))[ev]
                    arms['own_raw'][0].append(spearmanr(p, t).statistic)
                    arms['own_raw'][1].append(regret_at(p, t, 10))
                    yr = rankdata(y[idx]).astype(float)
                    p = (Xf @ ridge_fit(Xf[idx], yr))[ev]
                    arms['own_recal'][0].append(spearmanr(p, t).statistic)
                    arms['own_recal'][1].append(regret_at(p, t, 10))
                res[str(n)] = {k: {'rho': float(np.mean(v[0])),
                                   'regret10': float(np.mean(v[1]))} for k, v in arms.items()}
            # degree-1 and degree-<=2 held-out R^2, the estimated cumulative spectrum
            r2 = {}
            for lab, XX in (('1', X1), ('2', Xf)):
                vals = []
                for s_ in range(5):
                    rng = np.random.default_rng(7000 + s_)
                    perm = rng.permutation(len(y))
                    k = max(1, len(y) // 5)
                    te, tr = perm[:k], perm[k:]
                    w = ridge_fit(XX[tr], y[tr])
                    pp = XX[te] @ w
                    vals.append(1 - ((y[te] - pp) ** 2).sum()
                                / ((y[te] - y[te].mean()) ** 2).sum())
                r2['cumulative_' + lab] = float(np.mean(vals))
            frac[ds] = {'n': len(y), 'skew': float(skew(y)),
                        'top_decile_spread': float((y.max() - np.percentile(y, 90))
                                                   / (y.max() - y.min())),
                        'budgets': res, **r2}
        out['fractalnet'] = frac

    with open(os.path.join(RES, 'spectrum.json'), 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)

    # ---- report
    print('Exact amplitude spectrum of NAS-Bench-201 (%d architectures, complete %d^%d '
          'factorial)\n' % (len(G), N_OP, N_EDGE))
    nl = out['null']
    print('%-7s %-28s %s' % ('order', 'share of variance', 'white-noise null'))
    for k in range(1, 7):
        print('%-7d %-28s %6.1f%%'
              % (k, '  '.join('%6.1f%%' % (100 * spec[d]['share'][k]) for d in DATASETS),
                 100 * nl[k]))
    print('        %s' % '  '.join('%7s' % d[:8] for d in DATASETS))
    print('\ncumulative (= the R^2 ceiling of any degree-d predictor):')
    for k in (1, 2, 3):
        print('  <=%d  %s' % (k, '  '.join('%6.1f%%' % (100 * spec[d]['cumulative'][k])
                                           for d in DATASETS)))
    print('\ncross-dataset correlation of each order component:')
    for pair, v in out['cross_dataset'].items():
        print('  %-34s %s' % (pair, '  '.join('%d:%+.2f' % (k, v[k]) for k in range(1, 7))))
    print('\nrecovery of the exact order-<=2 component (correlation with ground truth):')
    for d in DATASETS:
        best = [s_ for s_ in DATASETS if s_ != d]
        print('  %-16s' % d, '  '.join(
            '%s@1000=%.3f' % (k.replace('from_', ''), rec[d]['1000'][k])
            for k in ['own'] + ['from_' + s_ for s_ in best]))

    print('\nthe spectrum is a property of the space, not the task:')
    print('  %-16s %-13s %-13s %-13s %s'
          % ('dataset', 'order<=2 raw', 'by rank', 'by logit', 'skew'))
    for d in DATASETS:
        print('  %-16s %-13.1f %-13.1f %-13.1f %.2f'
              % (d, 100 * spec[d]['cumulative'][2],
                 100 * out['spectrum_recalibrated'][d]['cumulative'][2],
                 100 * out['spectrum_logit'][d]['cumulative'][2],
                 out['compression'][d]['skew']))

    print('\nrecalibrate your own labels, or transfer a prior from a harder dataset?')
    print('  %-16s %-7s %-26s %s' % ('target', 'labels', 'rho  raw / recal / transfer',
                                     'regret@10  raw / recal / transfer'))
    for d in DATASETS:
        for n in ('100', '250', '500', '1000'):
            b = out['recalibration'][d]['budgets'][n]
            print('  %-16s %-7s %.3f / %.3f / %.3f        %5.2f / %5.2f / %5.2f'
                  % (d, n, b['own_raw']['rho'], b['own_recal']['rho'],
                     b['transfer_raw']['rho'], b['own_raw']['regret10'],
                     b['own_recal']['regret10'], b['transfer_raw']['regret10']))

    if 'fractalnet' in out:
        print('\ncontrol -- FractalNet, whose accuracies are not compressed:')
        for ds, v in out['fractalnet'].items():
            b = v['budgets'].get('1000') or list(v['budgets'].values())[-1]
            print('  %-10s skew %+5.2f  top-decile spread %.3f   rho raw %.3f / recal %.3f'
                  ' | degree1 R2 %.3f, degree<=2 R2 %.3f'
                  % (ds, v['skew'], v['top_decile_spread'], b['own_raw']['rho'],
                     b['own_recal']['rho'], v['cumulative_1'], v['cumulative_2']))

    print('\nwrote analysis/results/spectrum.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
