"""Figures for the new sections. Writes figures/fig_transfer, fig_sitl, fig_pareto (.pdf/.png).

Palette and rcParams are shared with s23_figures.py so the whole paper reads as one set.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common
from s23_figures import C, save, J, W1, W2
import matplotlib.pyplot as _plt
_plt.rcParams.update({'xtick.labelsize': 5.8, 'ytick.labelsize': 5.8})


def fig_transfer():
    t = J('transfer')
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.95), constrained_layout=True,
                             sharey=True)
    for ax, (key, tgt) in zip(axes, (('cifar10->cifar100', 'CIFAR-100'),
                                     ('cifar100->cifar10', 'CIFAR-10'))):
        e = t['dataset'][key]
        ns = sorted(int(k) for k in e['few_shot'])
        tr = [e['few_shot'][str(n)]['transfer'] for n in ns]
        sc = [e['few_shot'][str(n)]['scratch'] for n in ns]
        x = np.arange(len(ns))
        ax.axhline(e['epoch1_on_target'], color=C['grey'], ls=(0, (4, 3)), lw=0.9)
        ax.annotate('1 epoch', (0, e['epoch1_on_target']), textcoords='offset points',
                    xytext=(1, 3), ha='left', va='bottom', fontsize=5.4, color=C['grey'])
        m = [i for i, v in enumerate(sc) if v is not None]
        ax.plot(x[m], [sc[i] for i in m], 'o-', color=C['c100'], ms=3.2,
                label='fitted on target labels only')
        ax.plot(x, tr, 'o-', color=C['c10'], ms=3.2, label='transferred, then corrected')
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel('target labels', fontsize=6.4)
        ax.set_title(f'target: {tgt}', fontsize=6.6)
    axes[0].set_ylabel(r'Spearman $\rho$', fontsize=6.4)
    axes[0].set_ylim(0.38, 0.99)
    axes[0].legend(loc='lower right', fontsize=5.1)
    save(fig, 'fig_transfer')


def fig_sitl():
    s = J('sitl')
    budgets = [120, 260, 520]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.95), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(budgets))
    style = [('scratch', 'from scratch', C['c100'], 'o'),
             ('explore', '$+$ random exploration', C['alt'], 's'),
             ('transfer', '$+$ transferred prior', C['c10'], 'D')]
    for ds, ls in (('cifar10', '-'), ('cifar100', (0, (3, 2)))):
        for key, lab, col, mk in style:
            ax.plot(x, [s[ds][str(b)][key + '_delta'] for b in budgets], linestyle=ls,
                    color=col, marker=mk, ms=3.0,
                    label=(lab if ds == 'cifar10' else None))
    ax.axhline(0, color=C['grey'], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('architectures trained', fontsize=6.4)
    ax.set_ylabel('points vs training all', fontsize=6.4)
    ax.legend(loc='upper right', fontsize=5.1)
    ax.set_title('what the loop is worth', fontsize=6.6)

    ax = axes[1]
    w = 0.26
    for i, (key, lab, col, _) in enumerate(style):
        vals = [np.mean([s[ds][str(b)][key + '_fid'] for b in budgets])
                for ds in ('cifar10', 'cifar100')]
        ax.bar(np.arange(2) + (i - 1) * w, vals, w, color=col, label=lab)
    ax.axhline(s['cifar10']['prior_rho'], color=C['grey'], ls=(0, (4, 3)), lw=0.9)
    ax.annotate('prior', (-0.42, s['cifar10']['prior_rho']), textcoords='offset points',
                xytext=(0, 2), ha='left', fontsize=5.4, color=C['grey'])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['C-10', 'C-100'], fontsize=6.4)
    ax.set_ylabel(r'model $\rho$ at the end', fontsize=6.4)
    ax.set_ylim(0, 1.04)
    ax.set_title('why', fontsize=6.6)
    save(fig, 'fig_sitl')


def fig_pareto():
    sp = J('space')
    rows = list(csv.DictReader(open(os.path.join(common.DATA, 'cost_model.csv'))))
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.95), constrained_layout=True)
    COL = {'2': C['c10'], '3': C['alt'], '4': C['c100']}
    gt = {(r['uid'], r['dataset']): r for r in common.gene_table()
          if r['acc3'] is not None}
    for ax, ds in zip(axes, common.DATASETS):
        sub = [r for r in rows if r['dataset'] == ds]
        for nc in ('4', '3', '2'):
            xs, ys = [], []
            for r in sub:
                g = gt.get((r['uid'], ds))
                if g is None or str(g['n_columns']) != nc:
                    continue
                xs.append(float(r['params_M'])); ys.append(float(r['acc_epoch3_true']))
            ax.scatter(xs, ys, s=1.6, alpha=0.35, color=COL[nc], linewidths=0,
                       label=f'{nc} columns' if ds == 'cifar10' else None)
        fr = sp[ds]['pareto']
        ax.plot([p['params'] for p in fr], [p['acc'] for p in fr], color='black', lw=1.0,
                zorder=5)
        ax.set_xscale('log')
        ax.set_xlabel('parameters (M)', fontsize=6.4)
        ax.set_title(ds.upper().replace('CIFAR', 'CIFAR-'), fontsize=6.6)
    axes[0].set_ylabel('3-epoch accuracy (%)', fontsize=6.4)
    axes[0].legend(loc='lower right', fontsize=5.1, markerscale=4, handletextpad=0.2)
    save(fig, 'fig_pareto')


if __name__ == '__main__':
    fig_transfer()
    fig_sitl()
    fig_pareto()
