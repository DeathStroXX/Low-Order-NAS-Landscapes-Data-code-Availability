"""A measured noise floor, nine times over, on the same grid as the accuracy spectrum.

Section IV's concentration claim rests on a control: the high-order tail of the accuracy
spectrum is thin because structure is absent there, not because the estimator cannot resolve
what is there. The analytic white-noise null settles half of that -- noise would put 26 percent
of its variance at order six and the measurement puts 0.14 -- but it says nothing about this
benchmark's own measurement error. For that one needs a function on the *same* complete grid
whose high-order content is known to be noise.

HW-NAS-Bench (Li et al., ICLR 2021) supplies nine of them. It reports, for every one of the
15,625 NAS-Bench-201 cells, the measured latency on six hardware platforms (an Edge GPU, a
Raspberry Pi 4, an Edge TPU, a Pixel 3, the Eyeriss accelerator and an FPGA) and the measured
energy on three of them, plus Eyeriss's arithmetic intensity. Latency and energy are *sums over
the six edges up to measurement error and scheduling effects*: an edge's operation contributes
its own cost wherever it sits, so the analytic prediction is pure order one, and whatever an
exact decomposition finds above order one is this benchmark's own error on a quantity whose true
spectrum is known in advance.

That makes the nine columns three things at once:

  1. **Nine implementation certificates.** A decomposition that is correctly implemented must
     return order one for an additive function. Parameter and FLOP counts are additive by
     construction and return 100.000000 percent (Table~III); these are additive in physical
     reality instead, so what they return above order one is real and measurable.

  2. **Nine measured noise floors, on the accuracy grid.** The order-six share of each metric
     is the error floor of a measurement taken on the same 15,625 cells with the same estimator.
     Quoted against accuracy's 0.14 to 0.25 percent, it shows the estimator resolves noise when
     noise is present, which is the control the Limitations section previously only asserted.

  3. **A spread, not a single number.** The six platforms differ by orders of magnitude in how
     additive their latency is -- a compiled accelerator schedules a whole graph, a general
     processor executes it edge by edge -- so the floor comes with a range rather than a point
     estimate, which is the honest form of the control.

The table is the one its authors release, fetched once into `data/external/` and not modified.
It is read with a restricted unpickler: it is a pickle from outside this project, so only the
array and container types it legitimately contains are allowed to construct.

    python3 analysis/s58_hardware.py
"""
import csv
import io
import json
import os
import pickle
import re
import sys

import numpy as np

import anova

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
DATA = os.path.join(ROOT, 'data')
EXT = os.path.join(DATA, 'external')
HW = os.path.join(EXT, 'HW-NAS-Bench-v1_0.pickle')

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
SHAPE = (5,) * 6
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}
#: the measured columns, in the order the paper quotes them
METRICS = ['edgegpu_latency', 'edgegpu_energy', 'raspi4_latency', 'edgetpu_latency',
           'pixel3_latency', 'eyeriss_latency', 'eyeriss_energy', 'fpga_latency',
           'fpga_energy']
#: HW-NAS-Bench collects the first five columns by running each architecture on the device
#: itself and the last four from analytical cost models of an accelerator and an FPGA. The two
#: kinds answer different questions here, so they are never pooled: a device measurement carries
#: real measurement error and is the noise floor this paper quotes, while a cost model is
#: additive by construction and is an implementation certificate like the parameter count.
DEVICE_MEASURED = ['edgegpu_latency', 'edgegpu_energy', 'raspi4_latency', 'edgetpu_latency',
                   'pixel3_latency']
MODELLED = ['eyeriss_latency', 'eyeriss_energy', 'fpga_latency', 'fpga_energy']
PRETTY = {'edgegpu_latency': 'Edge GPU latency', 'edgegpu_energy': 'Edge GPU energy',
          'raspi4_latency': 'Raspberry Pi 4 latency', 'edgetpu_latency': 'Edge TPU latency',
          'pixel3_latency': 'Pixel 3 latency', 'eyeriss_latency': 'Eyeriss latency',
          'eyeriss_energy': 'Eyeriss energy', 'fpga_latency': 'FPGA latency',
          'fpga_energy': 'FPGA energy'}

#: Everything the released pickle legitimately needs in order to reconstruct. Anything else is
#: refused rather than executed: the file comes from outside this project.
_ALLOWED = {
    ('numpy', 'dtype'), ('numpy', 'ndarray'), ('numpy', 'float64'), ('numpy', 'int64'),
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('collections', 'OrderedDict'),
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError('refused to construct %s.%s from an external pickle'
                                     % (module, name))


def genotype(arch):
    toks = re.findall(r'([a-z0-9_]+)~\d', arch)
    if len(toks) != 6:
        raise ValueError('unexpected architecture string: %r' % arch)
    return [OPS.index(t) for t in toks]


def load_hw():
    if not os.path.exists(HW):
        print('HW-NAS-Bench table not found at %s' % HW)
        print('It is released by its authors and fetched, unmodified, with')
        print('  curl -L -o data/external/HW-NAS-Bench-v1_0.pickle \\')
        print('    https://raw.githubusercontent.com/GATECH-EIC/HW-NAS-Bench/main/'
              'HW-NAS-Bench-v1_0.pickle')
        return None
    with open(HW, 'rb') as fh:
        return _Restricted(fh).load()['nasbench201']


def accuracy():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'nb201_transfer.csv'))))
    G = np.array([genotype(r['arch']) for r in rows], dtype=int)
    acc = {d: np.array([float(r['acc_' + d]) for r in rows]) for d in DATASETS}
    return G, acc


def main():
    hw = load_hw()
    if hw is None:
        return 1
    Gacc, acc = accuracy()

    out = {'source': 'Li et al., HW-NAS-Bench: Hardware-Aware Neural Architecture Search '
                     'Benchmark, ICLR 2021',
           'n_arch': 15625, 'shape': list(SHAPE), 'metrics': METRICS,
           'accuracy_tail': {}, 'per_dataset': {}}

    for ds in DATASETS:
        a = anova.to_grid(Gacc, acc[ds], SHAPE)
        sp = anova.spectrum(a)
        out['accuracy_tail'][ds] = {'order1': sp[0], 'order6': sp[5],
                                    'order_le2': anova.cumulative(sp)[1]}

    for ds in DATASETS:
        tab = hw[ds]
        # The release keys its rows by the benchmark's own architecture string, so the
        # alignment is verified rather than assumed: decode each row's string to a genotype
        # and require the design to come out a complete 5^6 factorial exactly once over.
        arch = [c['arch_str'] for c in tab['config']]
        G = np.array([genotype(s) for s in arch], dtype=int)
        per = {}
        for m in METRICS:
            v = np.asarray(tab[m], dtype=float)
            if v.shape[0] != G.shape[0]:
                raise ValueError('%s/%s has %d rows for %d architectures'
                                 % (ds, m, v.shape[0], G.shape[0]))
            if not np.isfinite(v).all():
                raise ValueError('%s/%s contains non-finite values' % (ds, m))
            Y = anova.to_grid(G, v, SHAPE)
            anova.check_consistency(Y)
            sp = anova.spectrum(Y)
            per[m] = {'per_order': sp, 'order1': sp[0], 'order6': sp[5],
                      'above_order1': float(sum(sp[1:])),
                      'noise_ratio_vs_accuracy': sp[5] / out['accuracy_tail'][ds]['order6']}
        out['per_dataset'][ds] = per

    # ---- the summary the paper quotes --------------------------------------------------
    def gather(names, field):
        return [out['per_dataset'][ds][m][field] for ds in DATASETS for m in names]

    meas6, meas1 = gather(DEVICE_MEASURED, 'order6'), gather(DEVICE_MEASURED, 'order1')
    mod6, mod1 = gather(MODELLED, 'order6'), gather(MODELLED, 'order1')
    acc6 = [out['accuracy_tail'][ds]['order6'] for ds in DATASETS]
    out['kinds'] = {'device_measured': DEVICE_MEASURED, 'analytically_modelled': MODELLED}
    out['summary'] = {
        'n_device_columns': len(meas6), 'n_modelled_columns': len(mod6),
        'device_order6_min': float(min(meas6)), 'device_order6_max': float(max(meas6)),
        'device_order6_median': float(np.median(meas6)),
        'device_order1_min': float(min(meas1)), 'device_order1_max': float(max(meas1)),
        'modelled_order1_min': float(min(mod1)), 'modelled_order1_max': float(max(mod1)),
        'modelled_order6_max': float(max(mod6)),
        'modelled_exact_order1_columns': int(sum(1 for x in mod1 if x > 99.9999)),
        'accuracy_order6_min': float(min(acc6)), 'accuracy_order6_max': float(max(acc6)),
        'ratio_min': float(min(meas6) / max(acc6)),
        'ratio_max': float(max(meas6) / min(acc6)),
    }

    with open(os.path.join(RES, 'hardware.json'), 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print('HW-NAS-Bench on the NAS-Bench-201 grid: %d columns, each additive up to '
          'measurement error or by construction' % (3 * len(METRICS)))
    print()
    print('  %-24s %s' % ('metric', '   '.join('%-14s' % SHORT[d] for d in DATASETS)))
    print('  %-24s %s' % ('', '   '.join('%-14s' % 'order1 / o6' for d in DATASETS)))
    for m in METRICS:
        cells = []
        for ds in DATASETS:
            r = out['per_dataset'][ds][m]
            cells.append('%6.2f /%5.2f' % (r['order1'], r['order6']))
        print('  %-24s %s' % (PRETTY[m], '   '.join(cells)))
    print('  %-24s %s' % ('accuracy (reference)',
                          '   '.join('%6.2f /%5.2f'
                                     % (out['accuracy_tail'][d]['order1'],
                                        out['accuracy_tail'][d]['order6'])
                                     for d in DATASETS)))
    s = out['summary']
    print()
    print('  device-measured columns (%d): order-six share %.2f to %.2f%% (median %.2f), '
          'order-one share %.1f to %.1f%%'
          % (s['n_device_columns'], s['device_order6_min'], s['device_order6_max'],
             s['device_order6_median'], s['device_order1_min'], s['device_order1_max']))
    print('  modelled columns (%d): %d of them return order one to six decimals; the lowest '
          'is %.2f%% and no order-six share exceeds %.3f%%'
          % (s['n_modelled_columns'], s['modelled_exact_order1_columns'],
             s['modelled_order1_min'], s['modelled_order6_max']))
    print('  accuracy, same grid and same estimator: order six %.2f to %.2f%%'
          % (s['accuracy_order6_min'], s['accuracy_order6_max']))
    print('  so where a tail is present the estimator resolves one %.1f to %.0f times larger'
          % (s['ratio_min'], s['ratio_max']))
    print('\nwrote analysis/results/hardware.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
