"""Figure 2: the controls the exact reading needs, and the boundary it has.

Four panels, each drawn from a file in analysis/results/ so the figure carries exactly the
numbers the tables quote.

  (a) the concentration reproduces on a second complete factorial, and the null is measured.
      NAS-Bench-Macro's accuracy spectrum against two reference levels: the analytic
      white-noise null for a 3^8 grid, and the spectrum of the difference between two
      independent trainings of the same architectures -- a function on the same complete grid
      whose entire content is training noise. The measured and analytic nulls coincide, so the
      null is the right control and the accuracy tail is structure absent.

  (b) the estimator resolves a tail when one is there. The order-six share of every column
      HW-NAS-Bench measures on the NAS-Bench-201 grid, device measurements separated from
      analytic cost models, against accuracy's own order-six share on the same grid.

  (c) the ceiling does not order decisions. For each order d, the exact R^2 ceiling against the
      regret at a shortlist of ten of g_d, the squared-error-optimal degree-d function. Regret
      is not monotone in d, and recalibration takes the degree-2 projection to zero regret.

  (d) the scope boundary, diagnosed and repaired for nothing: held-out R^2 of a degree-2 fit on
      labels already spent, and what re-encoding by paths recovers.

    python3 analysis/s61_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common
from s23_figures import C, save, J, W1, W2

DS = [('cifar10', 'CIFAR-10', C['c10']),
      ('cifar100', 'CIFAR-100', C['c100']),
      ('ImageNet16-120', 'ImageNet16-120', C['good'])]


def _panel(ax, letter, title):
    ax.set_title('(%s) %s' % (letter, title), loc='left', pad=3.0)


def main():
    B, H, CE = J('benchmarks'), J('hardware'), J('ceiling')
    # Fig. 2 of the main paper is single-column and carries the two panels the argument
    # needs; the other two go to the supplement, which has the room for them.
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.72), constrained_layout=True)
    sm, smax = plt.subplots(1, 2, figsize=(W1, 1.72), constrained_layout=True)
    ax = [ax[0], smax[0], ax[1], smax[1]]

    # ---------------- (a) a second factorial, and a measured null ----------------
    a = ax[0]
    orders = np.arange(1, 9)
    a.plot(orders, B['accuracy']['per_order'], 'o-', ms=2.8, color=C['c10'],
           label='accuracy')
    a.plot(orders, B['measured_null']['per_order_mean'], 's-', ms=2.6, color=C['alt'],
           label='two trainings differenced')
    a.plot(orders, B['null']['per_order'], '^--', ms=2.4, color=C['grey'],
           label='white-noise null')
    a.set_yscale('log')
    a.set_xlabel('interaction order $d$')
    a.set_ylabel('% of variance')
    a.legend(loc='lower left', fontsize=5.0, handlelength=1.3, labelspacing=0.22,
             borderaxespad=0.1)
    a.set_xticks(orders)
    _panel(a, 'a', 'a second factorial')

    # ---------------- (b) the noise floor, measured, nine ways ----------------
    a = ax[1]
    dev = [H['per_dataset'][d][m]['order6']
           for d in H['accuracy_tail'] for m in H['kinds']['device_measured']]
    mod = [H['per_dataset'][d][m]['order6']
           for d in H['accuracy_tail'] for m in H['kinds']['analytically_modelled']]
    acc = [H['accuracy_tail'][d]['order6'] for d in H['accuracy_tail']]
    rng = np.random.default_rng(0)
    groups = [(mod, 'cost model', C['grey']), (acc, 'accuracy', C['c10']),
              (dev, 'measured on device', C['alt'])]
    for i, (v, lab, col) in enumerate(groups):
        v = np.asarray(v, dtype=float)
        v = np.clip(v, 1e-4, None)
        a.scatter(v, i + rng.uniform(-0.17, 0.17, size=v.size), s=7, color=col,
                  alpha=0.85, linewidths=0, label=lab)
    a.set_yticks(range(len(groups)))
    a.set_yticklabels([g[1] for g in groups], fontsize=6.0)
    a.set_xscale('log')
    a.set_xlabel('share at order 6 (%)')
    a.grid(axis='y', visible=False)
    _panel(a, 'a', 'the estimator sees noise')

    # ---------------- (c) the ceiling against the decision ----------------
    a = ax[2]
    for key, lab, col in DS:
        lad = CE['datasets'][key]['raw']['ladder']
        rec = CE['datasets'][key]['recalibrated']['ladder']
        a.plot([r['ceiling_r2_percent'] for r in lad], [r['regret_at_10'] for r in lad],
               'o-', ms=2.8, color=col, label=lab)
        a.plot([r['ceiling_r2_percent'] for r in rec], [r['regret_at_10'] for r in rec],
               ':', lw=1.0, color=col, alpha=0.85)
    a.plot([], [], ':', color=C['grey'], label='the same, recalibrated')
    a.axhline(0.1, color=C['grey'], lw=0.8, ls='--')
    a.text(26, 0.28, 'a tenth of a point', fontsize=5.0, color=C['grey'])
    a.set_xlabel('$R^{2}$ ceiling at order $\\leq d$ (%)')
    a.set_ylabel('regret of $g_d$ at ten (points)')
    a.legend(loc='upper center', fontsize=5.0, ncol=1, handlelength=1.3,
             borderaxespad=0.1, labelspacing=0.22)
    a.set_ylim(-0.35, 6.1)
    _panel(a, 'b', 'ceiling vs. decision')

    # ---------------- (d) the scope boundary ----------------
    a = ax[3]
    try:
        Z, E = J('zc_spaces'), J('encoding')
    except FileNotFoundError:
        Z = E = None
    if E is not None:
        #: the diagnostic is the held-out R^2 of a degree-2 fit on labels already spent; the
        #: order here puts the four spaces where it works above the one where it does not,
        #: with the path re-encoding of that one directly beneath it.
        want = [('nb201_ImageNet16-120_genotype', 'NB201 IN16'),
                ('nb201_cifar100_genotype', 'NB201 C100'),
                ('nb201_cifar10_genotype', 'NB201 C10'),
                ('nb301_genotype', 'NB301'),
                ('nb101_path', 'NB101, by paths'),
                ('nb101_genotype', 'NB101, adjacency')]
        held = E['diagnostic']
        names = [lab for k, lab in want if k in held]
        vals = [float(held[k]['test']) for k, _ in want if k in held]
        col = [C['good'] if v > 0 else C['c100'] for v in vals]
        a.barh(range(len(vals)), vals, color=col, height=0.62, linewidth=0)
        a.axvline(0, color=C['grey'], lw=0.8)
        a.set_yticks(range(len(vals)))
        a.set_yticklabels(names, fontsize=5.4)
        a.set_xlabel('held-out $R^{2}$, degree-2 fit')
        a.grid(axis='y', visible=False)
    _panel(a, 'b', 'the boundary is testable')

    save(fig, 'fig_controls')
    save(sm, 'fig_controls_sm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
