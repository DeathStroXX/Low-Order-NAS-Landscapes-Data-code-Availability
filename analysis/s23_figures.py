"""Shared figure style, and the two single-column figures of this paper.

The palette, the rcParams and the two column widths are imported by
s32_figures.py and s55_figures.py, so every figure in the paper reads as one set.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common

FIG = os.path.join(common.HERE, '..', 'figures')
# categorical palette, validated for colour-vision deficiency (see analysis/README_FIGURES)
C = {'c10': '#1f5fa8', 'c100': '#c9452c', 'good': '#1f8a70', 'alt': '#8a6bbf',
     'grey': '#6b6b6b', 'light': '#d8d8d4'}
plt.rcParams.update({
    'font.size': 7.2, 'axes.labelsize': 7.2, 'axes.titlesize': 7.6,
    'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6, 'legend.fontsize': 6.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'legend.frameon': False, 'figure.dpi': 160, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01, 'axes.grid': True, 'grid.linewidth': 0.4,
    'grid.color': '#e8e8e4', 'axes.axisbelow': True, 'pdf.fonttype': 42,
})
W1, W2 = 3.4, 7.1          # one- and two-column widths in inches


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'))
    plt.close(fig)
    print('wrote figures/' + name + '.pdf')


def J(n):
    return json.load(open(os.path.join(common.RES, n + '.json')))


# ---------------------------------------------------------------- fig 1: signals
def fig_signal():
    s = J('surrogate')
    fig, axes = plt.subplots(2, 1, figsize=(W1, 3.0),
                             gridspec_kw=dict(height_ratios=[1.0, 0.85]), constrained_layout=True)
    rows = [('Genotype surrogate', 'surrogate', 'free'),
            ('Surrogate $+$ one epoch', 'surrogate_plus_epoch1', '1 epoch'),
            ('One training epoch', 'epoch1', '1 epoch'),
            ('LLM accuracy predictor', 'predictor_llm', '1 call, post hoc')]
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    for i, (lab, k, cost) in enumerate(rows):
        a, b = s['cifar10']['rho'][k], s['cifar100']['rho'][k]
        ax.plot([min(a, b), max(a, b)], [y[i], y[i]], color=C['light'], lw=2.4,
                solid_capstyle='round', zorder=1)
        ax.scatter([a], [y[i]], color=C['c10'], s=30, zorder=3, edgecolor='white', lw=0.6)
        ax.scatter([b], [y[i]], color=C['c100'], s=30, zorder=3, edgecolor='white', lw=0.6)
        ax.annotate(f'{min(a, b):.3f}', (min(a, b), y[i]), textcoords='offset points',
                    xytext=(-6, -2.5), ha='right', fontsize=6.0, color=C['grey'])
        ax.annotate(f'{max(a, b):.3f}', (max(a, b), y[i]), textcoords='offset points',
                    xytext=(6, -2.5), ha='left', fontsize=6.0, color=C['grey'])
    ax.set_yticks(y); ax.set_yticklabels([f'{r[0]}\n({r[2]})' for r in rows], fontsize=5.9)
    ax.set_xlim(0.33, 1.06)
    ax.set_xlabel(r'Spearman $\rho$ against the 3-epoch reference')
    ax.scatter([], [], color=C['c10'], s=22, label='CIFAR-10')
    ax.scatter([], [], color=C['c100'], s=22, label='CIFAR-100')
    ax.legend(loc='upper left', ncol=1, handletextpad=0.2, borderpad=0.1)
    ax.set_title('(a) what each selection signal is worth', loc='left', fontsize=7.0)
    ax.grid(axis='y', visible=False)

    ax = axes[1]
    for ds, col, nm in (('cifar10', C['c10'], 'CIFAR-10'), ('cifar100', C['c100'], 'CIFAR-100')):
        e = s[ds]['sample_efficiency']
        xs = sorted(int(k) for k in e)
        ax.plot(xs, [e[str(k)][0] for k in xs], color=col, marker='o', markersize=3, label=nm)
        ax.axhline(s[ds]['rho']['epoch1'], color=col, ls=':', lw=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('architectures the surrogate was fitted on')
    ax.set_ylabel(r'held-out $\rho$')
    ax.set_ylim(0.54, 0.99)
    ax.annotate('one training epoch', (52, 0.955), fontsize=6.2, color=C['grey'])
    ax.legend(loc='lower right')
    ax.set_title('(b) how little data the surrogate needs', loc='left', fontsize=7.0)
    save(fig, 'fig_signal')


# ---------------------------------------------------------------- fig 2: budget
def fig_budget():
    p, h = J('prescreen'), J('halving')
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.88))
    ax = axes[0]
    Ms = [100, 200, 400, 800, 1600]
    for ds, col in (('cifar10', C['c10']), ('cifar100', C['c100'])):
        base = p[ds]['cells']['train_all']['acc']
        ys = [0.0] + [p[ds]['cells'][f'prescreen_{m}']['acc'] - base for m in Ms[1:]]
        ax.plot(Ms, ys, color=col, marker='o', markersize=3,
                label='CIFAR-10' if ds == 'cifar10' else 'CIFAR-100')
        ref = p[ds]['cells']['oracle3_fullcost']['acc'] - base
        ax.axhline(ref, color=col, ls=':', lw=0.9)
    ax.set_xscale('log'); ax.set_xticks(Ms); ax.set_xticklabels(Ms)
    ax.set_xlabel('candidates screened (100 of them then trained for one epoch)')
    ax.set_ylabel('gain in returned accuracy (pts)')
    ax.set_title('(a) pre-screening at a fixed 100-epoch budget', loc='left')
    ax.annotate('three epochs on all 100\n(3$\\times$ the training cost)', (430, 0.15),
                fontsize=6.2, color=C['grey'])
    ax.legend(loc='upper left')

    ax = axes[1]
    order = ['epoch1', 'halving_10', 'halving_25', 'halving_50', 'epoch2', 'full3']
    nice = {'epoch1': '1 ep', 'halving_10': 'halve 10%', 'halving_25': 'halve 25%',
            'halving_50': 'halve 50%', 'epoch2': '2 ep', 'full3': '3 ep'}
    for ds, col in (('cifar10', C['c10']), ('cifar100', C['c100'])):
        xs = [h[ds]['cells'][k]['frac'] * 100 for k in order]
        ys = [h[ds]['cells'][k]['regret'] for k in order]
        ax.plot(xs, ys, color=col, marker='o', markersize=3)
    offs = {'epoch1': (4, -2), 'halving_10': (-2, 7), 'halving_25': (-6, -11),
            'halving_50': (2, 7), 'epoch2': (4, 2), 'full3': (-14, 7)}
    for k in order:
        ax.annotate(nice[k], (h['cifar10']['cells'][k]['frac'] * 100,
                              h['cifar10']['cells'][k]['regret']),
                    textcoords='offset points', xytext=offs[k], fontsize=6.0, color=C['grey'])
    ax.set_xlabel('training cost (% of the three-epoch reference)')
    ax.set_ylabel('regret vs the reference (pts)')
    ax.set_title('(b) the budget frontier: halving dominates a flat epoch count', loc='left')
    save(fig, 'fig_budget')


if __name__ == '__main__':
    import sys
    want = sys.argv[1:] or ['signal', 'budget']
    for w in want:
        globals()['fig_' + w]()
