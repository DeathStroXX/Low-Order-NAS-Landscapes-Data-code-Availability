"""Multi-panel figures for the main paper (Figs. 2 and 3) and the supplement.

Every panel is drawn from a file in analysis/results/, so the figures carry exactly the
numbers the tables quote.  Palette, rcParams and the two column widths are shared with
s23_figures.py so the whole paper reads as one set.

  python3 analysis/s55_figures.py            # all figures
  python3 analysis/s55_figures.py spectrum_panels
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common
from s23_figures import C, save, J, W1, W2

DS = [('cifar10', 'CIFAR-10', C['c10']),
      ('cifar100', 'CIFAR-100', C['c100']),
      ('ImageNet16-120', 'ImageNet16-120', C['good'])]
ORD = [1, 2, 3, 4, 5, 6]


def _panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=3.0)


# ======================================================================== Fig. 2
def fig_spectrum_panels():
    """What the spectrum is, what it bounds, and what it buys.  Six panels."""
    s = J('spectrum')
    # Fig. 1 is single-column and carries panels (a)-(d); the two label-budget panels go to
    # the supplement, where they sit beside the table they illustrate.
    #: Fig. 1 spans both columns as a single row of four panels: the four read left to right
    #: as one argument, and a 2x2 block in one column forces the reader to track it in a Z.
    fig, axm = plt.subplots(1, 4, figsize=(W2, 1.85), constrained_layout=True)
    #: the label-efficiency panel is in the body (Fig. 3a); the supplementary figure carries
    #: the recalibration-against-transfer panel
    smfig, axs = plt.subplots(1, 1, figsize=(W1, 1.95), constrained_layout=True)
    ax = {(0, 0): axm[0], (0, 1): axm[1], (0, 2): axm[2],
          (1, 0): axm[3], (1, 2): axs}

    # (a) cumulative variance by interaction order, against the white-noise null
    a = ax[0, 0]
    null = [s['null'][str(o)] for o in ORD]
    cnull = np.cumsum(null)
    for key, lab, col in DS:
        cum = [s['spectrum'][key]['cumulative'][str(o)] * 100 for o in ORD]
        rec = [s['spectrum_recalibrated'][key]['cumulative'][str(o)] * 100 for o in ORD]
        a.plot(ORD, cum, 'o-', ms=2.6, color=col, label=lab)
        a.plot(ORD, rec, ':', lw=1.0, color=col, alpha=0.85)
    a.plot(ORD, cnull * 100, 's--', ms=2.4, color=C['grey'], label='white noise')
    a.plot([], [], ':', color=C['grey'], label='the same, recalibrated')
    a.axvline(2, color=C['light'], lw=4, zorder=0)
    a.set_xlabel('interaction order $\\leq d$')
    a.set_ylabel('cumulative variance (%)')
    a.set_ylim(0, 104)
    a.legend(loc='lower right', fontsize=5.6)
    _panel(a, 'a', 'spectrum is concentrated')

    # (b) how far above the null each order sits
    a = ax[0, 1]
    w, off = 0.26, -0.26
    for key, lab, col in DS:
        r = [s['null_ratio'][key][str(o)] for o in ORD]
        a.bar(np.array(ORD) + off, r, w, color=col, label=lab)
        off += 0.26
    a.axhline(1.0, color=C['grey'], lw=0.8, ls='--')
    a.set_yscale('log')
    a.set_xlabel('interaction order $d$')
    a.set_ylabel('variance / white-noise null')
    a.text(4.6, 1.5, 'null', fontsize=5.6, color=C['grey'])
    _panel(a, 'b', 'structure, not noise')

    # (c) the accuracy scale is compressed, and by how much
    a = ax[0, 2]
    x = np.arange(3)
    spread = [s['compression'][k]['top_decile_spread'] * 100 for k, _, _ in DS]
    gap = [s['compression'][k]['ceiling_gap'] for k, _, _ in DS]
    a.bar(x, spread, 0.56, color=[c for _, _, c in DS])
    for i, (sp, g) in enumerate(zip(spread, gap)):
        a.text(i, sp + 0.35, f'{g:.1f} pts\nto ceiling', ha='center', fontsize=5.0,
               color=C['grey'])
    fn = s['fractalnet']['cifar10']['top_decile_spread'] * 100
    a.axhline(fn, color=C['alt'], lw=1.0, ls=':')
    a.text(-0.42, fn + 0.4, 'FractalNet control', fontsize=5.2, color=C['alt'])
    a.set_xticks(x)
    a.set_xticklabels(['C-10', 'C-100', 'IN16'])
    a.set_ylim(0, 24)
    a.set_ylabel('best decile, % of accuracy range')
    _panel(a, 'c', 'easy tasks compressed')

    # (d) recalibrate, and the three spectra become one
    a = ax[1, 0]
    raw = [s['spectrum'][k]['cumulative']['2'] * 100 for k, _, _ in DS]
    rec = [s['spectrum_recalibrated'][k]['cumulative']['2'] * 100 for k, _, _ in DS]
    for i, (k, lab, col) in enumerate(DS):
        a.plot([0, 1], [raw[i], rec[i]], 'o-', ms=3.0, color=col, label=lab)
    fnr = s['fractalnet']['cifar10']['cumulative_2'] * 100
    a.plot([0, 1], [fnr, fnr], 'd:', ms=3.0, color=C['alt'], label='FractalNet (control)')
    a.set_xlim(-0.22, 1.30)
    a.set_ylim(64, 93)
    a.set_xticks([0, 1])
    a.set_xticklabels(['as measured', 'recalibrated'])
    a.set_ylabel('variance at order $\\leq 2$ (%)')
    a.legend(loc='lower right', fontsize=5.4)
    _panel(a, 'd', 'task-dependence is scale')

    # (f) recalibration against transfer, in the unit the decision is made in
    a = ax[1, 2]
    bud = [100, 250, 500, 1000]
    arms = [('own_raw', 'own labels', C['grey'], 0.55),
            ('transfer_raw', 'transfer from a harder task', C['alt'], 0.8),
            ('own_recal', 'own labels, recalibrated', C['good'], 1.0)]
    x = np.arange(len(bud))
    off = -0.27
    for k, lab, col, al in arms:
        v = [np.mean([s['recalibration'][d]['budgets'][str(b)][k]['regret10']
                      for d, _, _ in DS]) for b in bud]
        a.bar(x + off, v, 0.26, color=col, alpha=al, label=lab)
        off += 0.27
    a.set_xticks(x)
    a.set_xticklabels([str(b) for b in bud])
    a.set_xlabel('labels used to fit')
    a.set_ylabel('regret at a shortlist of ten (pts)')
    a.legend(loc='upper right', fontsize=5.2)
    a.set_title('what replaces transfer', loc='left', pad=3.0)
    save(fig, 'fig_spectrum_panels')
    save(smfig, 'fig_budget_panels')


# ======================================================================== Fig. 3
def fig_scope_panels():
    """Where the model holds, what it beats, and what a budget should buy.  Four panels."""
    e, z, rc, hv = J('encoding'), J('zc_nb201'), J('recipe'), J('halving')
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85), constrained_layout=True)

    # (a) the free diagnostic: held-out R^2 separates slot spaces from graph spaces
    a = ax[0]
    rows = [('NB-201 C-10', 'nb201_cifar10_genotype', C['c10']),
            ('NB-201 C-100', 'nb201_cifar100_genotype', C['c100']),
            ('NB-201 IN16', 'nb201_ImageNet16-120_genotype', C['good']),
            ('NB-301', 'nb301_genotype', C['alt']),
            ('NB-101', 'nb101_genotype', C['grey'])]
    y = np.arange(len(rows))[::-1]
    v = [e['diagnostic'][k]['test'] for _, k, _ in rows]
    a.barh(y, v, 0.62, color=[c for _, _, c in rows])
    a.axvline(0, color='k', lw=0.7)
    a.set_yticks(y)
    a.set_yticklabels([n for n, _, _ in rows], fontsize=5.8)
    a.set_xlabel('held-out $R^{2}$ of a degree-2 fit')
    a.text(0.015, 0.055, 'graph-structured', fontsize=5.2, color=C['grey'],
           transform=a.transAxes)
    a.text(0.015, 0.90, 'slot-structured', fontsize=5.2, color=C['grey'],
           transform=a.transAxes)
    _panel(a, 'a', 'the scope test is free')

    # (b) a re-encoding moves the same function to a lower order
    a = ax[1]
    bud = [100, 250, 500, 1000]
    for key, lab, col in (('genotype', 'operation slots', C['grey']),
                          ('path', 'input-to-output paths', C['good'])):
        a.plot(bud, [e['nb101']['rankers'][f'{key}|{b}']['rho'] for b in bud], 'o-',
               ms=2.6, color=col, label=lab)
    a.set_xscale('log')
    a.set_xlabel('labels used to fit (log)')
    a.set_ylabel('Spearman $\\rho$, NAS-Bench-101')
    a.legend(loc='lower right', fontsize=5.4)
    _panel(a, 'b', 'and it repairs the failure')

    # (c) the free ranker against the thirteen published zero-cost proxies
    a = ax[2]
    pd = z['per_dataset']['cifar10']
    pr = sorted(z['proxies'], key=lambda p: pd[f'proxy|{p}']['rho'])
    y = np.arange(len(pr))
    a.barh(y, [pd[f'proxy|{p}']['rho'] for p in pr], 0.68, color=C['light'])
    a.axvline(pd['surrogate|1000']['rho'], color=C['c10'], lw=1.1)
    a.axvline(pd['transfer|cifar100']['rho'], color=C['alt'], lw=1.1, ls='--')
    a.set_yticks(y)
    a.set_yticklabels([p.replace('_', ' ') for p in pr], fontsize=5.0)
    a.set_xlim(0.3, 1.0)
    a.set_xlabel('Spearman $\\rho$ (CIFAR-10)')
    a.text(pd['surrogate|1000']['rho'] - 0.015, 0.4, 'genotype, 1k labels', fontsize=5.0,
           color=C['c10'], rotation=90, va='bottom', ha='right')
    a.text(pd['transfer|cifar100']['rho'] + 0.012, 0.4, 'transferred, no label',
           fontsize=5.0, color=C['alt'], rotation=90, va='bottom')
    _panel(a, 'c', 'against the proxy family')

    # (d) what the same 100 training epochs buy under three policies
    a = ax[3]
    pol = [('B_screen800', 'free screening'),
           ('C_screen_halve', '$+$ halving'),
           ('ref_three_epochs_33', '3 epochs each,\n1/3 the candidates')]
    x = np.arange(len(pol))
    for i, (ds, lab, col) in enumerate([('cifar10', 'CIFAR-10', C['c10']),
                                        ('cifar100', 'CIFAR-100', C['c100'])]):
        base = rc[ds]['cells']['A_train100']['acc']
        v = [rc[ds]['cells'][k]['acc'] - base for k, _ in pol]
        err = [abs(rc[ds]['cells'][k]['acc'] - rc[ds]['cells'][k]['lo']) for k, _ in pol]
        a.bar(x + (i - 0.5) * 0.34, v, 0.32, yerr=err, color=col, label=lab,
              error_kw=dict(lw=0.6))
    a.axhline(0, color='k', lw=0.7)
    a.set_xticks(x)
    a.set_xticklabels([l for _, l in pol], fontsize=5.0, rotation=16, ha='right')
    a.set_ylabel('points against the flat policy')
    a.legend(loc='lower left', fontsize=5.4)
    _panel(a, 'd', 'what a budget should buy')
    save(fig, 'fig_scope_panels')





# ======================================================================== Fig. S1
def fig_policy_panels():
    """The three policy levers the body does not carry, on one scale.  Supplement."""
    si, wm, hl = J('sitl'), J('warmstart'), J('halving')
    #: the transfer panel is in the body (Fig. 3d); the three policy levers the body does not
    #: carry are drawn here, a third of the width each
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.85), constrained_layout=True)

    # (b) the same prior inside the search loop
    a = ax[0]
    bud = [120, 260, 520]
    x = np.arange(len(bud))
    arms = [('transfer', 'seeded with a prior', C['good']),
            ('explore', 'refitted, with exploration', C['alt']),
            ('scratch', 'refitted on what it measured', C['grey'])]
    off = -0.27
    for k, lab, col in arms:
        v = [np.mean([si[d][str(b)][k + '_delta'] for d in ('cifar10', 'cifar100')])
             for b in bud]
        a.bar(x + off, v, 0.26, color=col, label=lab)
        off += 0.27
    a.axhline(0, color='k', lw=0.7)
    a.set_xticks(x)
    a.set_xticklabels([str(b) for b in bud])
    a.set_xlabel('architectures trained')
    a.set_ylabel('points against training all')
    a.legend(loc='lower left', fontsize=5.0)
    _panel(a, 'a', 'a prior is the whole of it')

    # (b) a free warm start against a weak selection signal
    a = ax[1]
    rhos = ['0.33', '0.43', '0.58', '0.7', '0.93']
    for ds, lab, col in (('cifar10', 'CIFAR-10', C['c10']),
                         ('cifar100', 'CIFAR-100', C['c100'])):
        w = wm[ds]['asrun']
        a.plot([float(r) for r in rhos], [w[r]['loss'] for r in rhos], 'o--', ms=2.4,
               color=col, alpha=0.55)
        a.plot([float(r) for r in rhos], [w[r]['loss_warm'] for r in rhos], 'o-', ms=2.6,
               color=col, label=lab + ', warm start')
    a.axhline(0, color='k', lw=0.7)
    a.set_xlabel('fidelity $\\rho$ of the selection signal')
    a.set_ylabel('points lost against a clean signal')
    a.legend(loc='lower right', fontsize=5.2)
    _panel(a, 'b', 'a free warm start absorbs it')

    # (c) the halving frontier on the recovered curves
    a = ax[2]
    cells = [('epoch1', '1 epoch'), ('epoch2', '2 epochs'), ('full3', '3 epochs'),
             ('halving_25', 'halving 25%'), ('halving_50', 'halving 50%')]
    for ds, lab, col in (('cifar10', 'CIFAR-10', C['c10']),
                         ('cifar100', 'CIFAR-100', C['c100'])):
        c = hl[ds]['cells']
        xs = [c[k]['epochs'] / c['full3']['epochs'] for k, _ in cells]
        ys = [c[k]['regret'] for k, _ in cells]
        a.scatter(xs, ys, s=11, color=col, label=lab, zorder=3)
        offs = {'epoch1': (4, -1), 'epoch2': (4, 2), 'full3': (-4, 4),
                'halving_25': (2, 6), 'halving_50': (3, -6)}
        for (k, nm), xx, yy in zip(cells, xs, ys):
            if ds == 'cifar100':
                a.annotate(nm, (xx, yy), textcoords='offset points', xytext=offs[k],
                           fontsize=4.8, color=C['grey'],
                           ha='right' if k == 'full3' else 'left')
    a.set_xlabel('fraction of the full-fidelity budget')
    a.set_ylabel('regret against 3 epochs (pts)')
    a.legend(loc='upper right', fontsize=5.2)
    _panel(a, 'c', 'where halving sits')
    save(fig, 'fig_policy_panels')


if __name__ == '__main__':
    import sys
    want = sys.argv[1:] or ['spectrum_panels', 'scope_panels', 'policy_panels']
    for w in want:
        globals()['fig_' + w]()
