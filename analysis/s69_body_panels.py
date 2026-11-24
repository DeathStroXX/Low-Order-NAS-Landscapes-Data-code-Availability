"""The four-panel figure of the body (Fig. 3), at three layouts.

The paper uses the one-column 2x2 layout (fig_body_panels_1col.pdf); the two-panel and the
full-width row are written too.  Panels are drawn at the geometry of Figs. 1 and 2, so the
panel fonts are the ones the rest of the paper uses and nothing is rescaled.  Every number is read from analysis/results/, so the
panels carry exactly the quantities the tables quote.

  python3 analysis/s69_body_panels.py

Panels, and the result each reads:
  (a) results/spectrum.json    label efficiency                -> Section IV-C
  (b) results/zc_nb201.json    cost against fidelity           -> Section VI-A
  (c) results/regret.json      regret against shortlist size   -> Section VI-D
  (d) results/transfer.json    a prior against target labels   -> Section VII
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common
from s23_figures import C, save, J, W1, W2

DS = [('cifar10', 'CIFAR-10', C['c10']),
      ('cifar100', 'CIFAR-100', C['c100']),
      ('ImageNet16-120', 'ImageNet16-120', C['good'])]
KS_ZC = [1, 5, 10, 20, 50, 100, 200, 500]
NS = [100, 250, 500, 1000]


def _panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=3.0)


def fig_body_panels(layout='row'):
    """layout='row': one full-width row of four (figure*, cheapest per panel).
       layout='block': a single-column 2x2 block, for a build with no room for a figure*.
    """
    s, z, rg, tr = J('spectrum'), J('zc_nb201'), J('regret'), J('transfer')
    if layout == 'block':
        fig, axg = plt.subplots(2, 2, figsize=(W1, 3.05), constrained_layout=True)
        ax = [axg[0, 0], axg[0, 1], axg[1, 0], axg[1, 1]]
    elif layout == 'pair':
        #: Only the two panels Section VI argues from, one column wide. The other two stay in
        #: the supplement; this is the cheapest form of the figure that is still worth having.
        fig, axp = plt.subplots(1, 2, figsize=(W1, 1.62), constrained_layout=True)
        dump = plt.subplots(1, 2)[1]
        ax = [dump[0], axp[0], axp[1], dump[1]]
    else:
        fig, ax = plt.subplots(1, 4, figsize=(W2, 1.62), constrained_layout=True)

    # ---------------------------------------------------------------- (a)
    #: What a label budget buys, and what recalibrating the target's own labels adds to it.
    #: Dashed is the same budget without the correction, so the vertical gap between the two
    #: curves of a colour is the whole of what Section V recommends.
    a = ax[0]
    bud = [100, 250, 500, 1000]
    for key, lab, col in DS:
        b = s['recalibration'][key]['budgets']
        a.plot(bud, [b[str(x)]['own_recal']['rho'] for x in bud], 'o-', ms=2.6, color=col,
               label=lab)
        a.plot(bud, [b[str(x)]['own_raw']['rho'] for x in bud], 'o--', ms=2.2, color=col,
               alpha=0.5, lw=0.9)
    a.axvline(375, color=C['grey'], lw=0.8, ls=':')
    a.text(400, 0.86, '375\ncoefficients', fontsize=5.0, color=C['grey'], va='top')
    a.text(0.03, 0.97, 'dashed: not recalibrated', fontsize=5.0, color=C['grey'],
           va='top', transform=a.transAxes)
    a.set_xscale('log')
    a.set_xlabel('labels used to fit (log)')
    a.set_ylabel('Spearman $\\rho$')
    a.set_ylim(0.52, 0.99)
    a.legend(loc='lower right', fontsize=5.0)
    _panel(a, 'a', 'what a label budget buys')

    # ---------------------------------------------------------------- (b)
    #: Cost against fidelity. The point is the two orders of magnitude on the x axis, so only
    #: the proxies worth naming are labelled and the rest are plotted faint.
    a = ax[1]
    per_ds = z['per_dataset']
    dss = [k for k, _, _ in DS]
    NAMED = {'synflow': 'SynFlow', 'nwot': 'NWOT', 'zen': 'Zen', 'params': 'Params'}
    #: The named proxies sit within half a decade of one another, so a shared offset stacks
    #: their labels. These push each away from its own point without moving the anchor.
    OFF = {'nwot': (-3.0, 1.5), 'synflow': (-3.0, 2.0), 'params': (-3.0, -7.0),
           'zen': (3.0, -1.0)}
    proxies = sorted({k.split('|', 1)[1] for k in per_ds[dss[0]] if k.startswith('proxy|')})
    for m in proxies:
        x = np.mean([z['cost'][d][m]['sec_per_arch'] for d in dss])
        y = np.mean([per_ds[d]['proxy|' + m]['rho'] for d in dss])
        a.scatter(x, y, s=11, color=C['c100'], zorder=3,
                  alpha=1.0 if m in NAMED else 0.4)
        if m in NAMED:
            a.annotate(NAMED[m], (x, y), fontsize=5.0, xytext=OFF[m],
                       textcoords='offset points', color=C['c100'],
                       ha='left' if OFF[m][0] > 0 else 'right')
    xs = z['surrogate_cost']['sec_per_arch']
    ys = [np.mean([per_ds[d]['surrogate|%d' % n]['rho'] for d in dss]) for n in NS]
    a.plot([xs] * len(NS), ys, '-', color=C['c10'], lw=1.0, zorder=4)
    a.scatter([xs] * len(NS), ys, s=13, color=C['c10'], zorder=5)
    a.annotate('surrogate,\n100--1,000 labels', (xs, ys[0]), fontsize=5.0,
               xytext=(4.0, -9.0), textcoords='offset points', color=C['c10'])
    yt = np.mean([per_ds[d]['transfer|' + z['transfer_source_rule']['source'][d]]['rho']
                  for d in dss])
    a.scatter([xs], [yt], s=30, marker='*', color=C['good'], zorder=6)
    a.annotate('transferred in,\nno target label', (xs, yt), fontsize=5.0,
               xytext=(5.0, -1.0), textcoords='offset points', color=C['good'])
    a.set_xscale('log')
    a.set_xlim(xs / 7, 60)
    a.set_ylim(0.15, 0.95)
    a.set_xlabel('seconds to rank one (log)')
    a.set_ylabel('mean $\\rho$ over three datasets')
    _panel(a, 'a' if layout == 'pair' else 'b', 'what ranking costs')

    # ---------------------------------------------------------------- (c)
    #: Regret against shortlist size. CIFAR-10 solid, CIFAR-100 dashed; the dotted rule is
    #: the tenth of a point the text calls indistinguishable.
    a = ax[2]
    sty = [('surrogate', C['c10'], 'genotype surrogate'),
           ('transferred', C['good'], 'transferred'),
           ('epoch1', C['c100'], 'one measured epoch'),
           ('predictor', C['alt'], '7B predictor'),
           ('random', C['grey'], 'random shortlist')]
    for key, col, lab in sty:
        for ds, ls, mk in (('cifar10', '-', 'o'), ('cifar100', '--', 's')):
            v = [rg[ds]['regret'][key][str(k)]['mean'] for k in rg['ks']]
            a.plot(rg['ks'], v, ls, marker=mk, color=col, lw=1.1, ms=2.2,
                   label=lab if ds == 'cifar10' else None)
    a.axhline(0.1, color='k', ls=':', lw=0.7)
    a.set_xscale('log')
    a.set_yscale('symlog', linthresh=0.05)
    a.set_ylim(bottom=0.0)
    a.set_xlabel('architectures trained $k$ (log)')
    a.set_ylabel('regret (accuracy points)')
    a.legend(loc='lower left', fontsize=5.0, handlelength=1.4, labelspacing=0.2,
             borderpad=0.1, handletextpad=0.4)
    _panel(a, 'b' if layout == 'pair' else 'c', 'what a shortlist gives up')

    # ---------------------------------------------------------------- (d)
    #: A prior fitted on the other dataset against labels spent on the target. Dashed is the
    #: same model fitted on the target alone, so the horizontal gap is what the prior is worth
    #: in labels.
    a = ax[3]
    for key, lab, col in (('cifar10->cifar100', 'target CIFAR-100', C['c100']),
                          ('cifar100->cifar10', 'target CIFAR-10', C['c10'])):
        fs = tr['dataset'][key]['few_shot']
        bud = sorted((int(b) for b in fs))
        a.plot(bud, [fs[str(b)]['transfer'] for b in bud], 'o-', ms=2.4, color=col,
               label=lab)
        sc = [(b, fs[str(b)]['scratch']) for b in bud if fs[str(b)]['scratch'] is not None]
        a.plot([b for b, _ in sc], [v for _, v in sc], 'o--', ms=2.0, color=col, alpha=0.5)
    a.set_xscale('symlog', linthresh=25)
    a.set_xlabel('target labels (log)')
    a.set_ylabel('Spearman $\\rho$')
    a.text(0.03, 0.94, 'dashed: target alone', fontsize=5.0, color=C['grey'],
           transform=a.transAxes)
    a.legend(loc='center right', fontsize=5.0)
    _panel(a, 'd', 'what a prior is worth')

    save(fig, 'fig_body_panels' + {'block': '_1col', 'pair': '_2p'}.get(layout, ''))


if __name__ == '__main__':
    for lay in ('row', 'block', 'pair'):
        fig_body_panels(lay)
