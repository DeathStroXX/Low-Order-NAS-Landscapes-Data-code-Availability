"""Fig. 1: the whole paper as one procedure a practitioner can follow.

Every result in this manuscript answers one question in a sequence -- fit the free model,
check whether the encoding admits it, repair or fall back if it does not, fuse with a proxy
because the choice between them is not identifiable, then spend the training budget. Stated as
prose across six sections that sequence is invisible; drawn once it is the thing a reader
keeps. The numbers annotating each branch are read from `analysis/results/`, so the figure
cannot drift away from the tables.

    python3 analysis/s50_decision.py    # -> figures/fig_decision.pdf, .png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, '..', 'figures')

A, B = 'cifar10', 'cifar100'

INK = '#1a1a1a'
BOX = '#f4f4f2'
EDGE = '#5b5b5b'
GOOD = '#1f5c3a'
BAD = '#8c2f1d'
NOTE = '#4a4a4a'


def J(name):
    with open(os.path.join(RES, name + '.json')) as fh:
        return json.load(fh)


def numbers():
    """The quantities the figure annotates, all read from results/."""
    enc, spaces, zc = J('encoding'), J('zc_spaces'), J('zc_nb201')
    rec, sitl, reg = J('recipe'), J('sitl'), J('regret')
    diag = enc['diagnostic']
    nb201_tests = [diag[k]['test'] for k in diag if k.startswith('nb201')]
    return dict(
        r2_bad=diag['nb101_genotype']['test'],
        r2_good_lo=min(nb201_tests),
        r2_good_hi=max(nb201_tests),
        rho_nb101=spaces['nb101']['rankers']['surrogate|1000']['rho'],
        rho_path=enc['nb101']['rankers']['path|1000']['rho'],
        kstar=reg[A]['k_star']['surrogate'],
        kstar_b=reg[B]['k_star']['surrogate'],
        # the paired entries are differences in *regret*, where less is better, so the
        # advantage of the fusion is the negated difference -- the points it saves.
        fuse_a=-zc['paired'][A]['fused_picked_minus_pickedproxy']['mean'],
        fuse_b=-zc['paired'][B]['fused_picked_minus_pickedproxy']['mean'],
        recipe_a=(rec[A]['cells']['C_screen_halve']['acc']
                  - rec[A]['cells']['A_train100']['acc']),
        recipe_b=(rec[B]['cells']['C_screen_halve']['acc']
                  - rec[B]['cells']['A_train100']['acc']),
        sitl_a=sitl[A]['120']['transfer_delta'],
        sitl_b=sitl[B]['120']['transfer_delta'],
        scratch_a=sitl[A]['120']['scratch_delta'],
        scratch_b=sitl[B]['120']['scratch_delta'],
    )


#: geometry, in inches: the figure is laid out in real units and only then mapped to axes
#: coordinates, so a box is always tall enough for the lines it holds. The height is derived
#: from the content in `build`, so no box is clipped and no blank strip is left at the foot.
FIG_W, FIG_H = 3.45, 4.75
STANDFIRST_H = 0.13
PAD = 0.07              # outer margin
GAP = 0.165             # vertical gap between stacked boxes (room for the arrow)
TITLE_PT, BODY_PT = 7.2, 6.1
TITLE_H = TITLE_PT * 1.55 / 72.0
BODY_H = BODY_PT * 1.42 / 72.0
BOX_PAD = 0.055         # internal top/bottom padding


def _h(body):
    """Height in inches of a box holding a title and these body lines."""
    return TITLE_H + len(body) * BODY_H + 2 * BOX_PAD


def box(ax, x, y, w, h, title, body, fc=BOX, ec=EDGE, tc=INK):
    """`x, y, w, h` in inches, measured from the bottom-left of the figure."""
    X, Y, W, H = x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H
    ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle='round,pad=0,rounding_size=0.015',
                                linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
    top = y + h - BOX_PAD
    ax.text((x + w / 2.0) / FIG_W, top / FIG_H, title, ha='center', va='top',
            fontsize=TITLE_PT, weight='bold', color=tc, zorder=3)
    if body:
        ax.text((x + w / 2.0) / FIG_W, (top - TITLE_H) / FIG_H, '\n'.join(body),
                ha='center', va='top', fontsize=BODY_PT, color=NOTE, zorder=3,
                linespacing=1.42)


def arrow(ax, x, y0, y1, label=None, color=EDGE):
    """A downward arrow from y0 to y1 at x, all in inches."""
    ax.add_patch(FancyArrowPatch((x / FIG_W, y0 / FIG_H), (x / FIG_W, y1 / FIG_H),
                                 arrowstyle='-|>', mutation_scale=7, linewidth=0.9,
                                 color=color, zorder=1, shrinkA=0, shrinkB=0))
    if label:
        ax.text(x / FIG_W + 0.035, ((y0 + y1) / 2.0) / FIG_H, label, ha='left', va='center',
                fontsize=6.1, color=color, weight='bold', zorder=3)


def build(n):
    global FIG_H
    full = FIG_W - 2 * PAD
    half = (full - 0.10) / 2.0

    steps = [
        ('Fit the genotype model on labels you have',
         ['ridge on gene indicators and their pairs; a few',
          'labels per gene-pair cell, pooled over runs']),
        (u'Read its held-out $R^{2}$ \u2014 free, no new label',
         ['slot-structured genotype: $+%.2f$ to $+%.2f$' % (n['r2_good_lo'], n['r2_good_hi']),
          'graph genotype: $%.2f$, worse than the mean' % n['r2_bad']]),
        (None, None),                       # the in-scope / out-of-scope pair
        ('Fuse with a zero-cost proxy, 2:1, untuned',
         ['which proxy is best is not identifiable without',
          'target labels, so do not choose between them',
          '$+%.2f$ and $+%.2f$ points over an honest choice'
          % (n['fuse_a'], n['fuse_b'])]),
        ('Spend the training budget',
         ['screen free, then allocate epochs by halving:',
          '$+%.2f$ and $+%.2f$ points at an unchanged budget'
          % (n['recipe_a'], n['recipe_b']),
          'architectures left to train: $k^{\\star}=%s$ and $%s$'
          % (n['kstar'], n['kstar_b'])]),
        ('Only now, a model-in-the-loop search',
         ['seeded with the transferred prior: $+%.2f$, $+%.2f$'
          % (n['sitl_a'], n['sitl_b']),
          'refitted on what it chose to measure: $%.2f$, $%.2f$'
          % (n['scratch_a'], n['scratch_b'])]),
    ]
    branch = [('In scope', ['use it as the', 'first-stage ranker'], '#eef5ef', GOOD, GOOD),
              ('Out of scope', ['re-encode by paths:',
                                '$\\rho\\;%.2f\\!\\rightarrow\\!%.2f$'
                                % (n['rho_nb101'], n['rho_path'])], '#f7eeec', BAD, BAD)]

    heights = [_h(b[1]) if b[0] else max(_h(c[1]) for c in branch) for b in steps]
    total = sum(heights) + GAP * (len(steps) - 1)
    FIG_H = total + 2 * PAD + STANDFIRST_H
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    y = PAD
    tops = []
    for h in reversed(heights):
        tops.append((y, h))
        y += h + GAP
    tops.reverse()

    for (title, body), (yy, hh) in zip(steps, tops):
        if title is None:
            box(ax, PAD, yy, half, hh, branch[0][0], branch[0][1],
                fc=branch[0][2], ec=branch[0][3], tc=branch[0][4])
            box(ax, PAD + half + 0.10, yy, half, hh, branch[1][0], branch[1][1],
                fc=branch[1][2], ec=branch[1][3], tc=branch[1][4])
        else:
            box(ax, PAD, yy, full, hh, title, body)

    mid = PAD + full / 2.0
    lx, rx = PAD + half / 2.0, PAD + half + 0.10 + half / 2.0
    for i in range(len(steps) - 1):
        y_from = tops[i][0]
        y_to = tops[i + 1][0] + tops[i + 1][1]
        if i == 1:                                   # the branch
            arrow(ax, lx, y_from, y_to, 'yes', GOOD)
            arrow(ax, rx, y_from, y_to, 'no', BAD)
        elif i == 2:                                 # both branches rejoin
            arrow(ax, lx, y_from, y_to)
            arrow(ax, rx, y_from, y_to)
        else:
            arrow(ax, mid, y_from, y_to)

    ax.text(0.5, (FIG_H - PAD) / FIG_H, 'every test on this path is free',
            ha='center', va='top', fontsize=6.5, style='italic', color=NOTE)
    return fig


def main():
    n = numbers()
    fig = build(n)
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig_decision.' + ext), dpi=400)
    plt.close(fig)
    with open(os.path.join(RES, 'decision.json'), 'w') as fh:
        json.dump(n, fh, indent=1, sort_keys=True)
    print('wrote figures/fig_decision.pdf/.png and results/decision.json')
    for k, v in sorted(n.items()):
        print('  %-12s %s' % (k, v))
    return 0


if __name__ == '__main__':
    sys.exit(main())
