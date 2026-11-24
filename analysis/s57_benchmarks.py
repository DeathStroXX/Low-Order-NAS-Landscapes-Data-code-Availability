"""The exact spectrum on a second public NAS benchmark, and a *measured* white-noise null.

Section IV computes the amplitude spectrum of NAS-Bench-201 exactly, because that space is a
complete factorial. It is not the only public NAS benchmark that is one, and a single landscape
cannot say whether the concentration it shows is a property of NAS spaces or of that space. This
script answers the question on a second, independent benchmark of a different shape.

**NAS-Bench-Macro** (Su et al., CVPR 2021) is a macro-level space: eight searchable layers, each
carrying one of three blocks (identity, MB3_K3, MB6_K5), so the space is a complete 3^8 = 6{,}561
factorial, and every architecture in it was *trained from scratch three independent times* on
CIFAR-10. It differs from NAS-Bench-201 in every respect that could matter here -- macro rather
than cell-level, depth rather than operation choice, eight ternary factors rather than six
five-level ones, a different training pipeline and a different research group -- which is what
makes it a test rather than a replication.

Three things come out of it, all exact:

  1. **The concentration reproduces.** The spectrum is computed by the same Moebius inversion,
     with no model fitted, and the order-<=2 and order-<=3 shares are reported against the
     white-noise null for a 3^8 grid, which is C(8,k) 2^k / (3^8 - 1) per order and is a
     different null from NAS-Bench-201's.

  2. **The null is measured, not only derived.** Each architecture was trained three times, so
     the difference between two seeds is a function on the *same complete grid* whose only
     content is training noise. Decomposing it is therefore a direct measurement of where this
     benchmark's own noise sits, with no assumption and no model: if the analytic null is the
     right control, the seed-difference spectrum must match it. This is the strongest form of
     the control that Section IV's claim needs, and NAS-Bench-201 cannot supply it because its
     table reports seed means rather than per-seed accuracies.

  3. **Cost is order-1 here too.** Parameter and FLOP counts are sums over the eight layers by
     construction, so an exact decomposition must return pure order one on this grid as well.
     It is the same implementation certificate as Table~III's, on a grid of a different shape,
     which is what rules out an accident of 5^6.

The scale argument of Section VI is also re-run here: skewness, the top-decile spread, the
power-family dose-response and the rank recalibration, so that the compression mechanism is
tested on a landscape whose accuracy distribution was produced by a different pipeline.

The benchmark table is the one its authors release, fetched once into `data/external/` and not
modified; `PROVENANCE.md` there records where it came from and its license.

    python3 analysis/s57_benchmarks.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, skew, spearmanr

import anova

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
EXT = os.path.join(ROOT, 'data', 'external')

MACRO = os.path.join(EXT, 'nas-bench-macro_cifar10.json')
#: eight searchable layers, three candidate blocks each
MACRO_SHAPE = (3,) * 8
BLOCKS = ['Identity', 'MB3_K3', 'MB6_K5']
#: the power family used for the dose-response of Section VI, as in s56
POWERS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0]


def load_macro():
    """NAS-Bench-Macro as a (G, per-seed accuracies, params, flops) tuple on the 3^8 grid."""
    if not os.path.exists(MACRO):
        print('NAS-Bench-Macro table not found at %s' % MACRO)
        print('It is released by its authors and fetched, unmodified, with')
        print('  curl -L -o data/external/nas-bench-macro_cifar10.json \\')
        print('    https://raw.githubusercontent.com/xiusu/NAS-Bench-Macro/master/'
              'data/nas-bench-macro_cifar10.json')
        return None
    with open(MACRO) as fh:
        tab = json.load(fh)
    keys = sorted(tab)
    G = np.array([[int(c) for c in k] for k in keys], dtype=int)
    if G.shape != (3 ** 8, 8):
        raise ValueError('expected a complete 3^8 design, got %s' % (G.shape,))
    if G.min() != 0 or G.max() != 2:
        raise ValueError('expected ternary genes, got range %d..%d' % (G.min(), G.max()))
    seeds = np.array([tab[k]['test_acc'] for k in keys], dtype=float)
    mean = np.array([tab[k]['mean_acc'] for k in keys], dtype=float)
    if not np.allclose(seeds.mean(axis=1), mean, atol=1e-6):
        raise ValueError('mean_acc disagrees with the per-seed accuracies it averages')
    params = np.array([tab[k]['params'] for k in keys], dtype=float)
    flops = np.array([tab[k]['flops'] for k in keys], dtype=float)
    return G, seeds, mean, params, flops


def spec(G, y, shape):
    Y = anova.to_grid(G, y, shape)
    anova.check_consistency(Y)
    sp = anova.spectrum(Y)
    return sp, anova.cumulative(sp), Y


def compression(y):
    """How compressed a response is: skewness, and the range the best decile occupies."""
    y = np.asarray(y, dtype=float)
    lo, hi = y.min(), y.max()
    rng = hi - lo
    thr = np.quantile(y, 0.9)
    top = y[y >= thr]
    return {'skewness': float(skew(y)),
            'top_decile_spread': float((top.max() - top.min()) / rng) if rng else 0.0,
            'range': float(rng), 'mean': float(y.mean()), 'std': float(y.std())}


def decision_metrics(score, truth, k=10):
    """Rank fidelity, regret at a shortlist of k, and k* for a ranking of the whole grid."""
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    rho = float(spearmanr(score, truth).statistic)
    best = truth.max()
    order = np.argsort(-score, kind='stable')
    reg = float(best - truth[order[:k]].max())
    kstar = None
    for kk in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        if best - truth[order[:kk]].max() < 0.1:
            kstar = kk
            break
    return {'rho': rho, 'regret_at_%d' % k: reg, 'kstar': kstar}


def main():
    loaded = load_macro()
    if loaded is None:
        return 1
    G, seeds, acc, params, flops = loaded
    shape = MACRO_SHAPE
    out = {'benchmark': 'NAS-Bench-Macro',
           'source': 'Su et al., Prioritized Architecture Sampling with Monte-Carlo Tree '
                     'Search, CVPR 2021',
           'dataset': 'cifar10', 'n_arch': int(G.shape[0]), 'n_factors': int(G.shape[1]),
           'n_levels': 3, 'blocks': BLOCKS, 'n_seeds': int(seeds.shape[1])}

    null = anova.null_spectrum(shape)
    out['null'] = {'per_order': null, 'cumulative': anova.cumulative(null)}

    # ---------------- 1. the accuracy spectrum, exactly ----------------
    sp, cum, Yacc = spec(G, acc, shape)
    out['accuracy'] = {'per_order': sp, 'cumulative': cum,
                       'compression': compression(acc),
                       'null_ratio': [sp[i] / null[i] for i in range(len(sp))]}

    # ---------------- 2. the measured null: a seed difference ----------------
    # Two independent trainings of the same architecture differ only by training noise, so
    # their difference is a function on the same complete grid whose entire content is noise.
    # Its spectrum is this benchmark's own noise floor, measured rather than assumed.
    diffs = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        dsp, dcum, _ = spec(G, seeds[:, a] - seeds[:, b], shape)
        diffs.append(dsp)
    diffs = np.array(diffs)
    out['measured_null'] = {
        'pairs': [[0, 1], [0, 2], [1, 2]],
        'per_order_mean': diffs.mean(axis=0).tolist(),
        'per_order_std': diffs.std(axis=0).tolist(),
        'cumulative_mean': anova.cumulative(diffs.mean(axis=0).tolist()),
        'max_abs_deviation_from_analytic_null': float(
            np.abs(diffs.mean(axis=0) - np.array(null)).max()),
        'seed_var_share_of_total': float(
            seeds.var(axis=1, ddof=1).mean() / acc.var()),
    }
    # per-seed spectra: the structure must not depend on which training run is read
    per_seed = [spec(G, seeds[:, s], shape)[0] for s in range(seeds.shape[1])]
    out['per_seed'] = {'per_order': [list(map(float, p)) for p in per_seed],
                       'cumulative_order2': [float(anova.cumulative(p)[1]) for p in per_seed]}

    # ---------------- 3. cost is exactly additive ----------------
    out['certificates'] = {}
    for name, v in [('params', params), ('flops', flops)]:
        s, c, _ = spec(G, v, shape)
        out['certificates'][name] = {'per_order': s, 'order1': s[0],
                                     'above_order1': float(sum(s[1:]))}

    # ---------------- 4. the ceiling, and what a degree-2 ridge of it is worth -------------
    sums = anova.order_sums(Yacc)
    idx = tuple(G.T)
    ceil, dec = [], []
    g = np.zeros(shape)
    for d in range(1, len(shape) + 1):
        g = g + sums[d]
        ceil.append(cum[d - 1])
        dec.append(decision_metrics(g[idx], acc))
    out['ceiling'] = {'r2_percent': ceil, 'decision': dec}

    # ---------------- 5. the scale mechanism, on this landscape ----------------
    rk = rankdata(acc)
    rsp, rcum, _ = spec(G, rk, shape)
    out['recalibrated'] = {'per_order': rsp, 'cumulative': rcum}
    dose = []
    y0 = (acc - acc.min()) / (acc.max() - acc.min())
    for p in POWERS:
        yp = y0 ** p
        s, c, _ = spec(G, yp, shape)
        dose.append({'p': p, 'skewness': float(skew(yp)), 'order_le2': c[1]})
    out['dose_response'] = dose

    with open(os.path.join(RES, 'benchmarks.json'), 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    # ---------------- report ----------------
    print('NAS-Bench-Macro: complete 3^%d factorial, %d architectures, %d seeds each'
          % (G.shape[1], G.shape[0], seeds.shape[1]))
    print()
    print('  order        1      2      3      4      5      6      7      8')
    def row(lab, v, f='%6.2f'):
        print(('  %-10s' % lab) + ''.join(f % x for x in v))
    row('accuracy', sp)
    row('cumulative', cum)
    row('seed diff', diffs.mean(axis=0))
    row('null', null)
    print()
    print('  order <=2 carries %.1f%% of accuracy variance against a null of %.1f%%'
          % (cum[1], out['null']['cumulative'][1]))
    print('  order <=3 carries %.1f%%  (null %.1f%%)'
          % (cum[2], out['null']['cumulative'][2]))
    print('  the measured null (seed differences) sits within %.2f points of the analytic null'
          % out['measured_null']['max_abs_deviation_from_analytic_null'])
    print('  seed variance is %.3f%% of the accuracy variance'
          % (100 * out['measured_null']['seed_var_share_of_total']))
    print('  params: order 1 = %.6f%%, everything above it = %.3g%%'
          % (out['certificates']['params']['order1'],
             out['certificates']['params']['above_order1']))
    print('  flops : order 1 = %.6f%%, everything above it = %.3g%%'
          % (out['certificates']['flops']['order1'],
             out['certificates']['flops']['above_order1']))
    print('  skewness %.2f, top-decile spread %.3f'
          % (out['accuracy']['compression']['skewness'],
             out['accuracy']['compression']['top_decile_spread']))
    print('  after rank recalibration order <=2 is %.1f%%' % rcum[1])
    print('  degree-2 predictor: R^2 ceiling %.1f%%, attainable rho %.3f, regret@10 %.2f, '
          'k* %s' % (ceil[1], dec[1]['rho'], dec[1]['regret_at_10'], dec[1]['kstar']))
    print('\nwrote analysis/results/benchmarks.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
