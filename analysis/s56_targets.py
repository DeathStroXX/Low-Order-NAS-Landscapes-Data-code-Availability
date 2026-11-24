"""What else the complete factorial decomposes, and what each answer certifies.

NAS-Bench-201 tabulates more than test accuracy on its 5^6 grid: it also tabulates parameter
count, FLOPs, measured latency, training time, training accuracy and cross-entropy loss. Every
one of those is a function on the *same* complete factorial, so every one admits the same exact
decomposition -- and each answers a question the accuracy spectrum alone cannot.

  1. certificate   Parameter count and FLOPs are sums over the six edges by construction, so
                   an exact decomposition MUST place 100% of their variance at order one and
                   nothing anywhere else. It does, to 1e-22 of a percent. That is not a result
                   about NAS; it is a correctness proof of the implementation on a target whose
                   true spectrum is known a priori, and it is the control a reader should want
                   before believing the accuracy spectrum.

  2. noise         The white-noise null in s53 is analytic: it says where variance would sit if
                   the landscape carried no structure. It does not say what this benchmark's own
                   measurement noise looks like. Latency does. Latency is very nearly additive
                   over edges, so its true high-order content is near zero and whatever the
                   decomposition finds at order six is measurement noise -- and it finds 6x to
                   25x more of it there than in accuracy, on the same grid with the same
                   estimator. Accuracy's thin tail is therefore not the estimator failing to
                   resolve noise; the estimator resolves noise fine when noise is present.

  3. cost          The cost objectives are exactly order one and the accuracy objective is not.
                   Multi-objective NAS is thus not two problems of the same kind: one objective
                   is identified by six marginal means and needs no search at all, and the
                   entire difficulty sits in the other. It also explains, structurally, why
                   parameter and FLOP counts work as zero-cost proxies at all.

  4. scale         The dose-response that makes Section VI's mechanism quantitative. Taking
                   CIFAR-10 alone and pushing it through a power family -- one landscape, one
                   task, nothing changing but the scale -- traces a curve of order-<=2 share
                   against skewness. The three real tasks lie ON that curve: their order-<=2
                   shares are predicted from CIFAR-10's own scale curve to a fraction of a
                   point, across a spread of twenty. The apparent task-dependence is therefore
                   reproducible without changing the task.

  5. anatomy       Where the order-one mass actually sits. One edge carries two-fifths of it and
                   another carries a fortieth, and the ORDERING of the six is the same on every
                   task even though the shares are not -- which is what a transferred prior
                   actually transfers, and why it transfers at all.

  6. realization   The ceiling is only useful if someone says how close a real model gets. The
                   free degree-2 ridge realizes 85-95% of the degree-2 ceiling, so the margin
                   available to an elaborate predictor is that remainder plus whatever lies
                   above order two -- which the spectrum also bounds.

    python3 analysis/s56_targets.py
"""
import csv
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)

import s53_spectrum as S  # noqa: E402  (the exact decomposition lives there; we do not re-derive it)

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
#: the six edges of the cell, in the order s53's genotype() emits them
EDGE_NAMES = ['1<-0', '2<-0', '2<-1', '3<-0', '3<-1', '3<-2']
#: exponents of the power family. p<1 compresses the top of the range further, p>1 expands it.
POWERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
#: held-out R^2 of the degree-2 genotype ridge, as reported in tables/tab_zc_main.tex
HELD_OUT_R2 = {'cifar10': 0.58, 'cifar100': 0.70, 'ImageNet16-120': 0.84}

PROBLEMS = []


def skew(y):
    """Sample skewness. Written out so the script does not need SciPy."""
    y = np.asarray(y, float)
    m = y - y.mean()
    s = np.sqrt((m ** 2).mean())
    return float((m ** 3).mean() / s ** 3)


def top_decile_range(y):
    """Share of the response range occupied by the best tenth -- the compression statistic."""
    y = np.asarray(y, float)
    return float((y.max() - np.quantile(y, 0.9)) / (y.max() - y.min()))


def spectrum_of(G, y):
    """Variance share and cumulative share by interaction order, exactly."""
    comp, Y = S.anova(G, np.asarray(y, float))
    _, var = S.by_order(comp, Y.shape)
    tot = sum(var.values())
    return (comp, Y,
            {k: var[k] / tot for k in var},
            {k: sum(var[j] for j in range(1, k + 1)) / tot for k in var})


def main():
    G, acc = S.load()
    archs = [r['arch'] for r in csv.DictReader(
        open(os.path.join(ROOT, 'data', 'nb201_transfer.csv')))]
    tbl = pickle.load(open(os.path.join(ROOT, 'data', 'nb201_all.pickle'), 'rb'))

    def column(ds, field):
        y = np.array([tbl[a][ds].get(field, np.nan) for a in archs], float)
        bad = int(np.isnan(y).sum())
        if bad:
            y[np.isnan(y)] = np.nanmean(y)
        return y, bad

    out = {'n_arch': len(archs), 'datasets': DATASETS, 'powers': POWERS}

    # ---------------- 1. the exactness certificate ----------------
    cert = {}
    for field in ('params', 'flop'):
        y, _ = column('cifar10', field)
        _, _, share, _ = spectrum_of(G, y)
        cert[field] = {'order1': share[1], 'orders_2_to_6': sum(share[k] for k in range(2, 7))}
        if abs(share[1] - 1.0) > 1e-9:
            PROBLEMS.append('%s is additive over edges by construction, so order 1 must carry '
                            '100%% of its variance; got %.6f%%' % (field, 100 * share[1]))
    out['certificate'] = cert

    # ---------------- 2/3. cost targets, and latency as a measured noise reference ----------
    cost = {}
    for ds in DATASETS:
        cost[ds] = {}
        for field in ('params', 'flop', 'latency', 'train_times'):
            y, bad = column(ds, field)
            _, _, share, cum = spectrum_of(G, y)
            cost[ds][field] = {'order1': cum[1], 'order6': share[6], 'imputed': bad}
        _, _, ashare, acum = spectrum_of(G, acc[ds])
        cost[ds]['accuracy'] = {'order1': acum[1], 'order6': ashare[6], 'imputed': 0}
        cost[ds]['latency_noise_ratio'] = (cost[ds]['latency']['order6']
                                           / cost[ds]['accuracy']['order6'])
    out['cost'] = cost

    # ---------------- 4. the scale curve, and whether the three tasks lie on it ------------
    y0 = acc['cifar10']
    z = (y0 - y0.min()) / (y0.max() - y0.min())
    curve = []
    for p in POWERS:
        yy = z ** p
        _, _, _, cum = spectrum_of(G, yy)
        curve.append({'p': p, 'skew': skew(yy), 'top_decile_range': top_decile_range(yy),
                      'order_le2': cum[2], 'order_le3': cum[3]})
    out['scale_curve'] = curve

    measured, collapse = {}, {}
    for ds in DATASETS:
        _, _, _, cum = spectrum_of(G, acc[ds])
        measured[ds] = {'skew': skew(acc[ds]), 'top_decile_range': top_decile_range(acc[ds]),
                        'order_le2': cum[2]}
    xs = np.array([c['skew'] for c in curve])
    ys = np.array([c['order_le2'] for c in curve])
    o = np.argsort(xs)
    for ds in DATASETS:
        pred = float(np.interp(measured[ds]['skew'], xs[o], ys[o]))
        collapse[ds] = {'measured': measured[ds]['order_le2'], 'predicted': pred,
                        'abs_error': abs(pred - measured[ds]['order_le2'])}
    spread = max(m['order_le2'] for m in measured.values()) - \
             min(m['order_le2'] for m in measured.values())
    out['measured'] = measured
    out['collapse'] = {'per_dataset': collapse,
                       'mae': float(np.mean([c['abs_error'] for c in collapse.values()])),
                       'spread_explained': float(spread)}
    # A scale change on one landscape must not merely move the spectrum, it must move it to
    # where the other tasks actually are; if it cannot, the mechanism claim is not supported.
    if out['collapse']['mae'] > 0.1 * spread:
        PROBLEMS.append('the CIFAR-10 scale curve predicts the other tasks to %.2f points, '
                        'more than a tenth of the %.2f-point spread it claims to explain'
                        % (100 * out['collapse']['mae'], 100 * spread))

    # a fourth landscape, free: the validation split is a different protocol on the same grid
    yv, _ = column('cifar10-valid', 'eval_acc1es')
    _, _, _, cumv = spectrum_of(G, yv)
    out['cifar10_valid'] = {'skew': skew(yv), 'order_le2': cumv[2]}

    # the natural alternative scale, reported because it does NOT work
    loss_le2 = {}
    for ds in DATASETS:
        y, bad = column(ds, 'eval_losses')
        _, _, _, cum = spectrum_of(G, y)
        loss_le2[ds] = {'order_le2': cum[2], 'imputed': bad}
    out['loss'] = {'per_dataset': loss_le2,
                   'spread': max(v['order_le2'] for v in loss_le2.values())
                             - min(v['order_le2'] for v in loss_le2.values()),
                   'accuracy_spread': float(spread)}

    # the generalization gap, which is harder than either accuracy it is built from
    gap = {}
    for ds in DATASETS:
        tr, _ = column(ds, 'train_acc1es')
        ev, _ = column(ds, 'eval_acc1es')
        _, _, gshare, gcum = spectrum_of(G, tr - ev)
        _, _, eshare, _ = spectrum_of(G, ev)
        gap[ds] = {'order_le2': gcum[2], 'order6': gshare[6], 'eval_order6': eshare[6]}
    out['gap'] = gap

    # ---------------- 5. where the order-one mass sits ----------------
    anatomy = {}
    for ds in DATASETS:
        comp, Y, _, _ = spectrum_of(G, acc[ds])
        v = [float((np.broadcast_to(comp[(i,)], Y.shape) ** 2).mean()) for i in range(6)]
        t = sum(v)
        anatomy[ds] = {'share': {EDGE_NAMES[i]: v[i] / t for i in range(6)},
                       'rank': [EDGE_NAMES[i] for i in np.argsort(v)[::-1]]}
    # the edge means, which say which operation the main effect prefers
    op_means = {ds: {EDGE_NAMES[i]: {S.OPS[o]: float(acc[ds][G[:, i] == o].mean())
                                     for o in range(len(S.OPS))}
                     for i in range(6)} for ds in DATASETS}
    out['anatomy'] = anatomy
    out['edge_op_means'] = op_means
    orders = [tuple(anatomy[ds]['rank']) for ds in DATASETS]
    out['anatomy_rank_invariant'] = len(set(orders)) == 1

    # ---------------- 6. how much of the ceiling the free model realizes ----------------
    ceiling = json.load(open(os.path.join(HERE, 'results', 'spectrum.json')))['ceiling']
    real = {}
    for ds in DATASETS:
        c = ceiling[ds]['2']
        real[ds] = {'ceiling_degree2': c, 'held_out_r2': HELD_OUT_R2[ds],
                    'realized': HELD_OUT_R2[ds] / c}
        if not 0.0 < real[ds]['realized'] <= 1.05:
            PROBLEMS.append('%s: held-out R^2 %.2f against a degree-2 ceiling of %.3f is not a '
                            'sane realization fraction' % (ds, HELD_OUT_R2[ds], c))
    out['realization'] = real

    # ---------------- report ----------------
    print('the other functions on the same complete factorial (%d cells)\n' % len(archs))
    print('1. exactness certificate -- a target whose spectrum is known before we measure it')
    for f, c in cert.items():
        print('   %-8s order 1 carries %.6f%% of variance; orders 2-6 carry %.3e%%'
              % (f, 100 * c['order1'], 100 * c['orders_2_to_6']))

    print('\n2/3. cost is additive, accuracy is not; and latency measures this grid\'s noise')
    print('   %-15s %8s %8s %9s' % ('dataset', 'lat ord1', 'lat ord6', 'acc ord6'))
    for ds in DATASETS:
        c = cost[ds]
        print('   %-15s %7.1f%% %7.3f%% %8.3f%%   -> %4.1fx more order-6 than accuracy'
              % (ds, 100 * c['latency']['order1'], 100 * c['latency']['order6'],
                 100 * c['accuracy']['order6'], c['latency_noise_ratio']))

    print('\n4. the scale curve on CIFAR-10 alone, and the three tasks measured against it')
    for c in curve:
        print('   p=%-5g skew %7.3f  top decile %5.1f%% of range  order<=2 %6.2f%%'
              % (c['p'], c['skew'], 100 * c['top_decile_range'], 100 * c['order_le2']))
    print('   predicted from that one landscape\'s scale curve alone:')
    for ds in DATASETS:
        c = collapse[ds]
        print('   %-15s measured %6.2f%%   predicted %6.2f%%   error %4.2f points'
              % (ds, 100 * c['measured'], 100 * c['predicted'], 100 * c['abs_error']))
    print('   mean absolute error %.2f points, against the %.1f-point spread it explains'
          % (100 * out['collapse']['mae'], 100 * spread))
    print('   cross-entropy loss, for contrast: spread %.1f points (accuracy %.1f) -- a '
          'natural\n   scale does NOT remove the ordering, which is why a rank map is used'
          % (100 * out['loss']['spread'], 100 * spread))

    print('\n5. where the order-one mass sits')
    for ds in DATASETS:
        print('   %-15s %s' % (ds, '  '.join(
            '%s %4.1f%%' % (e, 100 * anatomy[ds]['share'][e]) for e in EDGE_NAMES)))
    print('   ordering identical on all three tasks: %s' % out['anatomy_rank_invariant'])

    print('\n6. what the free degree-2 ridge realizes of its own ceiling')
    for ds in DATASETS:
        r = real[ds]
        print('   %-15s held-out R^2 %+.2f against a ceiling of %.3f  ->  %.1f%% realized'
              % (ds, r['held_out_r2'], r['ceiling_degree2'], 100 * r['realized']))

    if PROBLEMS:
        print('\nFAILED:')
        for p in PROBLEMS:
            print('  - %s' % p)
    else:
        print('\nall internal checks passed')

    # ---------------- the table the manuscript uses ----------------
    L = []
    L.append(r'\begin{table}[!t]')
    L.append(r'\centering')
    L.append(r'\caption{Other functions on the same complete factorial. '
             r'\textbf{(a)} Parameter and FLOP counts are additive over the six edges by '
             r'construction, so their spectrum is known before it is measured: recovering it '
             r'is a correctness check, not a finding. \textbf{(b)} Latency is additive up to '
             r'measurement error, so its order-6 share is this grid\textquotesingle s own '
             r'noise floor---$6$ to $25$ times what accuracy shows there. \textbf{(c)} '
             r'CIFAR-10 alone, pushed through $z\mapsto z^{p}$ on the accuracy rescaled to '
             r'$[0,1]$: the predicted column is read '
             r'off that one landscape\textquotesingle s curve at each task\textquotesingle s '
             r'skewness. \textbf{(d)} What the free degree-2 ridge realizes of its ceiling. '
             r'Produced by \texttt{analysis/s56\_targets.py}.}')
    L.append(r'\label{tab:targets}')
    L.append(r'\footnotesize')
    L.append(r'\setlength{\tabcolsep}{4pt}')
    L.append(r'\begin{tabular}{@{}lrrr@{}}')
    L.append(r'\toprule')
    L.append(r'\multicolumn{4}{@{}l}{\emph{(a) additive by construction, and the certificate}}'
             r' \\')
    L.append(r'target & order 1 & orders 2--6 & \\')
    L.append(r'\midrule')
    for f, lab in (('params', 'parameter count'), ('flop', 'FLOPs')):
        L.append(r'%s & $%.4f\%%$ & $%.0e\%%$ & \\'
                 % (lab, 100 * cert[f]['order1'], 100 * cert[f]['orders_2_to_6']))
    L.append(r'training time & $%.1f\%%$ & $%.1f\%%$ & \\'
             % (100 * cost['cifar10']['train_times']['order1'],
                100 * (1 - cost['cifar10']['train_times']['order1'])))
    L.append(r'\addlinespace[2pt]')
    L.append(r'\multicolumn{4}{@{}l}{\emph{(b) latency as a measured noise floor}} \\')
    L.append(r'dataset & lat.\ order 1 & lat.\ order 6 & acc.\ order 6 \\')
    L.append(r'\midrule')
    for ds in DATASETS:
        c = cost[ds]
        L.append(r'%s & $%.1f\%%$ & $%.3f\%%$ & $%.3f\%%$ \\'
                 % (ds.replace('_', r'\_'), 100 * c['latency']['order1'],
                    100 * c['latency']['order6'], 100 * c['accuracy']['order6']))
    L.append(r'\addlinespace[2pt]')
    L.append(r'\multicolumn{4}{@{}l}{\emph{(c) the three tasks against CIFAR-10\textquotesingle s '
             r'own scale curve}} \\')
    L.append(r'dataset & skewness & order $\le 2$ & predicted \\')
    L.append(r'\midrule')
    for ds in DATASETS:
        L.append(r'%s & $%.2f$ & $%.1f\%%$ & $%.1f\%%$ \\'
                 % (ds, measured[ds]['skew'], 100 * collapse[ds]['measured'],
                    100 * collapse[ds]['predicted']))
    L.append(r'\addlinespace[2pt]')
    L.append(r'\multicolumn{4}{@{}l}{\emph{(d) realization of the degree-2 ceiling}} \\')
    L.append(r'dataset & ceiling & held-out $R^{2}$ & realized \\')
    L.append(r'\midrule')
    for ds in DATASETS:
        r = real[ds]
        L.append(r'%s & $%.3f$ & $%+.2f$ & $%.0f\%%$ \\'
                 % (ds, r['ceiling_degree2'], r['held_out_r2'], 100 * r['realized']))
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    tdest = os.path.join(ROOT, 'tables', 'tab_targets.tex')
    open(tdest, 'w').write('\n'.join(L) + '\n')
    print('wrote tables/tab_targets.tex')

    dest = os.path.join(HERE, 'results', 'targets.json')
    json.dump(out, open(dest, 'w'), indent=1, sort_keys=True, default=float)
    print('wrote analysis/results/targets.json')
    return 1 if PROBLEMS else 0


if __name__ == '__main__':
    sys.exit(main())
