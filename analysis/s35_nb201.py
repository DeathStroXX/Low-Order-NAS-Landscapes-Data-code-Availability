"""External validation on NAS-Bench-201: is the free genotype surrogate a property of
discrete NAS spaces, or of our space?

Everything the paper claims about the ranking signal is re-run here on a public tabular
benchmark the authors of this paper did not design, over three datasets rather than two, one
of which (ImageNet16-120) differs from the others in input resolution, class count and task
difficulty.

The benchmark table is `data/nb201_all.pickle`: the 15,625 cells of the NAS-Bench-201
topology space, each with final test accuracy, parameter count, FLOPs and measured latency
for cifar10, cifar100 and ImageNet16-120. A cell is six categorical genes -- the six edges
of the cell, each one of five operations -- so the surrogate of `surrogate.py` applies with
nothing changed but the cardinalities.

Outputs
-------
results/nb201.json         every number below
../tables/tab_nb201.tex    Table: fidelity, transfer and the direction rule
../figures/fig_nb201.pdf   label efficiency and the transfer matrix
"""
import itertools, json, os, pickle, re, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau

import common

DS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}
OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OI = {o: i for i, o in enumerate(OPS)}
NG = 6
CARD = np.full(NG, len(OPS), dtype=int)
OFF = np.concatenate([[0], np.cumsum(CARD)[:-1]])
NMAIN = int(CARD.sum())
PAIRS = list(itertools.combinations(range(NG), 2))
_po, _p = [], NMAIN
for _a, _b in PAIRS:
    _po.append(_p)
    _p += int(CARD[_a] * CARD[_b])
PAIR_OFF = np.array(_po)
NFEAT = _p
GRID = int(len(OPS) ** NG)
#: the identifiability argument of Section IV, for this space
N_CELLS = int(sum(CARD[a] * CARD[b] for a, b in PAIRS))


def encode(arch_str):
    """'|op~0|+|op~0|op~1|+|...' -> the six operation codes, in edge order."""
    toks = re.findall(r'([a-z_0-9x]+)~\d', arch_str)
    return [OI[t] for t in toks]


def design(P, pairs=True):
    P = np.asarray(P, int)
    n = len(P)
    A = np.zeros((n, NFEAT if pairs else NMAIN), dtype=np.float64)
    r = np.arange(n)
    A[r[:, None], OFF[None, :] + P] = 1.0
    if pairs:
        for pi, (a, b) in enumerate(PAIRS):
            A[r, PAIR_OFF[pi] + P[:, a] * CARD[b] + P[:, b]] = 1.0
    return A


def fit(P, y, alpha=10.0, pairs=True):
    """The ridge of surrogate.py, in the normal equations. Returns a predict function."""
    A = design(P, pairs)
    y = np.asarray(y, float)
    b = float(y.mean())
    G = A.T @ A
    G[np.diag_indices_from(G)] += alpha
    w = np.linalg.solve(G, A.T @ (y - b))
    return lambda Q: design(Q, pairs) @ w + b


def regret(pred, y, k):
    """Accuracy points lost by training the k best-predicted candidates instead of all.

    The sort is stable, so cells a predictor ties (parameter count and FLOPs tie often on
    this space) keep the table's order and the result does not depend on the NumPy version.
    """
    o = np.argsort(-pred, kind='stable')[:k]
    return float(y.max() - y[o].max())


def r2(pred, y):
    y = np.asarray(y, float)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def load():
    f = os.path.join(common.DATA, 'nb201_all.pickle')
    if not os.path.exists(f):
        sys.exit('data/nb201_all.pickle is missing. It is the NAS-Bench-201 accuracy table '
                 'as redistributed by NASLib (naslib/data/nb201_all.pickle in the automl/'
                 'NASLib repository, 9.5 MB): all 15,625 cells with final test accuracy, '
                 'parameters, FLOPs and measured latency on cifar10, cifar100 and '
                 'ImageNet16-120. Place it in data/ and rerun.')
    with open(f, 'rb') as fh:
        tab = pickle.load(fh)
    keys = list(tab)
    P = np.array([encode(k) for k in keys], dtype=np.int16)
    y = {d: np.array([tab[k][d]['eval_acc1es'] for k in keys], float) for d in DS}
    cost = {'params': np.array([tab[keys[0]][DS[0]]['params']] * 0, float)}
    cost = {
        'params': np.array([tab[k][DS[0]]['params'] for k in keys], float),
        'flops': np.array([tab[k][DS[0]]['flop'] for k in keys], float),
        'latency': np.array([tab[k][DS[0]]['latency'] for k in keys], float),
    }
    return keys, P, y, cost


def main():
    keys, P, Y, COST = load()
    n = len(keys)
    out = {'n_arch': n, 'grid': GRID, 'n_pair_cells': N_CELLS, 'datasets': DS,
           'n_genes': NG, 'n_ops': len(OPS), 'n_features': NFEAT}
    assert n == GRID == 15625, (n, GRID)

    # ---------------------------------------------------------------- label efficiency
    # Fit on N labels, test on every architecture held out.  5 draws, fixed seeds.
    NS = [100, 250, 500, 1000]
    NSEED = 20                 # every split-based quantity in this file uses these 20 splits
    out['fidelity'] = {}
    for d in DS:
        y = Y[d]
        out['fidelity'][d] = {}
        for N in NS:
            rho, tau, rg1, rg20, add = [], [], [], [], []
            for s in range(NSEED):
                idx = np.random.default_rng(s).permutation(n)
                tr, te = idx[:N], idx[N:]
                f = fit(P[tr], y[tr])
                p = f(P[te])
                rho.append(spearmanr(p, y[te]).correlation)
                tau.append(kendalltau(p, y[te]).correlation)
                rg1.append(regret(p, y[te], 1))
                rg20.append(regret(p, y[te], 20))
                fa = fit(P[tr], y[tr], pairs=False)
                add.append(spearmanr(fa(P[te]), y[te]).correlation)
            out['fidelity'][d][str(N)] = {
                'rho': float(np.mean(rho)), 'rho_sd': float(np.std(rho)),
                'tau': float(np.mean(tau)),
                'regret1': float(np.mean(rg1)), 'regret20': float(np.mean(rg20)),
                'rho_additive': float(np.mean(add))}
        # the two trivial zero-cost proxies that come free with the benchmark table
        out['fidelity'][d]['params'] = {
            'rho': float(spearmanr(COST['params'], y).correlation),
            'tau': float(kendalltau(COST['params'], y).correlation),
            'regret20': regret(COST['params'], y, 20)}
        out['fidelity'][d]['flops'] = {
            'rho': float(spearmanr(COST['flops'], y).correlation),
            'tau': float(kendalltau(COST['flops'], y).correlation),
            'regret20': regret(COST['flops'], y, 20)}
        out['fidelity'][d]['best'] = float(y.max())
        out['fidelity'][d]['mean'] = float(y.mean())

    # ------------------------------- one pass over shared splits: transfer, regret, random
    # Everything below is computed on the SAME 20 random splits, so the own-fitted model, the
    # transferred model and a random shortlist are compared on identical held-out pools and no
    # quantity in this file has two values.
    NFIT = 1000
    KS = [1, 5, 10, 20, 50, 100, 200, 500]
    acc_tr = {f'{s_}->{t}': {'rho': [], 'tau': [], 'rg': []}
              for s_ in DS for t in DS if s_ != t}
    acc_own = {d: {'rho': [], 'rg': {k: [] for k in KS}} for d in DS}
    acc_rnd = {d: {k: [] for k in KS} for d in DS}
    for sd in range(NSEED):
        g = np.random.default_rng(sd)
        idx = g.permutation(n)
        tr, te = idx[:NFIT], idx[NFIT:]
        models = {d: fit(P[tr], Y[d][tr]) for d in DS}
        preds = {d: models[d](P[te]) for d in DS}
        noise = g.standard_normal(len(te))
        for d in DS:
            yt = Y[d][te]
            acc_own[d]['rho'].append(spearmanr(preds[d], yt).correlation)
            for k in KS:
                acc_own[d]['rg'][k].append(regret(preds[d], yt, k))
                acc_rnd[d][k].append(regret(noise, yt, k))
            for src in DS:
                if src == d:
                    continue
                e = acc_tr[f'{src}->{d}']
                e['rho'].append(spearmanr(preds[src], yt).correlation)
                e['tau'].append(kendalltau(preds[src], yt).correlation)
                e['rg'].append(regret(preds[src], yt, 20))

    out['n_seeds'] = NSEED
    out['n_fit'] = NFIT
    out['regret'] = {}
    out['regret_random'] = {}
    for d in DS:
        out['regret'][d] = {str(k): {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                                     'max': float(np.max(v))}
                            for k, v in acc_own[d]['rg'].items()}
        out['regret_random'][d] = {str(k): {'mean': float(np.mean(v))}
                                   for k, v in acc_rnd[d].items()}
        ks10 = [k for k in KS if out['regret'][d][str(k)]['mean'] <= 0.1]
        ks1 = [k for k in KS if out['regret'][d][str(k)]['mean'] <= 1.0]
        out['regret'][d]['k_star_0.1'] = (min(ks10) if ks10 else None)
        out['regret'][d]['k_star_1.0'] = (min(ks1) if ks1 else None)
        # the own-fitted fidelity on these same splits, so the transfer table's "own" column
        # and the label-efficiency curve above are the same quantity
        out['fidelity'][d]['own_1000_shared'] = float(np.mean(acc_own[d]['rho']))

    out['transfer'] = {}
    for key, e in acc_tr.items():
        tgt = key.split('->')[1]
        out['transfer'][key] = {
            'rho': float(np.mean(e['rho'])), 'rho_sd': float(np.std(e['rho'])),
            'tau': float(np.mean(e['tau'])),
            'regret20': float(np.mean(e['rg'])),
            'rho_fitted_here': float(np.mean(acc_own[tgt]['rho'])),
            'regret20_fitted_here': float(np.mean(acc_own[tgt]['rg'][20])),
            'regret20_random': float(np.mean(acc_rnd[tgt][20])),
            'beats_own': bool(np.mean(e['rho']) > np.mean(acc_own[tgt]['rho']))}

    # the direction rule: transfer out of the harder dataset against transfer into it.
    order = sorted(DS, key=lambda d: Y[d].mean())          # hardest first
    hard, easy = order[0], order[-1]
    out['direction'] = {
        'hardest': hard, 'easiest': easy,
        'mean_acc': {d: float(Y[d].mean()) for d in DS},
        'down': float(np.mean([out['transfer'][f'{hard}->{t}']['rho']
                               for t in DS if t != hard])),
        'up': float(np.mean([out['transfer'][f'{easy}->{t}']['rho']
                             for t in DS if t != easy])),
        'beats_own_down': int(sum(out['transfer'][f'{s_}->{t}']['beats_own']
                                  for s_ in DS for t in DS
                                  if s_ != t and Y[s_].mean() < Y[t].mean())),
        'n_down': int(sum(1 for s_ in DS for t in DS
                          if s_ != t and Y[s_].mean() < Y[t].mean()))}

    # ------------------------------------------------------------------ cost-constrained
    # One surrogate, fitted once, used inside every deployment budget. Same 20 splits, and a
    # random shortlist of the same size inside each band, for the same reason as above.
    out['bands'] = {}
    QS = [0.25, 0.5, 0.75, 1.0]
    for d in DS:
        y = Y[d]
        acc = {m: {str(q): {'rho': [], 'rg': [], 'rnd': [], 'n': 0} for q in QS}
               for m in ['params', 'flops', 'latency']}
        for sd in range(NSEED):
            g = np.random.default_rng(sd)
            idx = g.permutation(n)
            tr, te = idx[:NFIT], idx[NFIT:]
            p = fit(P[tr], y[tr])(P[te])
            noise = g.standard_normal(len(te))
            for metric in acc:
                c = COST[metric][te]
                for q in QS:
                    m = c <= np.quantile(c, q)
                    e = acc[metric][str(q)]
                    e['rho'].append(spearmanr(p[m], y[te][m]).correlation)
                    e['rg'].append(regret(p[m], y[te][m], 20))
                    e['rnd'].append(regret(noise[m], y[te][m], 20))
                    e['n'] = int(m.sum())
        out['bands'][d] = {
            metric: {q: {'n': e['n'], 'rho': float(np.mean(e['rho'])),
                         'regret20': float(np.mean(e['rg'])),
                         'regret20_random': float(np.mean(e['rnd']))}
                     for q, e in bands.items()}
            for metric, bands in acc.items()}

    emit_csv(keys, P, Y, COST, NFIT)
    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'nb201.json'), 'w'), indent=1)
    emit_table(out)
    emit_figure(out, P, Y)
    print(json.dumps({'fidelity_1000': {d: round(out['fidelity'][d]['1000']['rho'], 3)
                                        for d in DS},
                      'transfer': {k: round(v['rho'], 3)
                                   for k, v in out['transfer'].items()},
                      'direction': {'down': round(out['direction']['down'], 3),
                                    'up': round(out['direction']['up'], 3)},
                      'regret20': {d: round(out['regret'][d]['20']['mean'], 3) for d in DS}},
                     indent=1))


def emit_csv(keys, P, Y, COST, nfit):
    """One row per NAS-Bench-201 cell: the measured accuracy on each dataset, the surrogate's
    held-out estimate fitted on that dataset, and the estimate transferred from each of the
    other two with no label of the target. Released so the transfer result can be rechecked
    without refitting anything."""
    import csv
    idx = np.random.default_rng(1).permutation(len(keys))
    tr = idx[:nfit]
    fitted = {d: fit(P[tr], Y[d][tr]) for d in DS}
    pred = {d: fitted[d](P) for d in DS}
    held = np.zeros(len(keys), bool)
    held[idx[nfit:]] = True
    path = os.path.join(common.DATA, 'nb201_transfer.csv')
    with open(path, 'w', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n')
        wr.writerow(['arch', 'held_out', 'params_M', 'flops_M', 'latency_s']
                    + ['acc_' + d for d in DS]
                    + ['own_' + d for d in DS]
                    + ['from_%s_to_%s' % (s_, t) for t in DS for s_ in DS if s_ != t])
        for i, k in enumerate(keys):
            row = [k, int(held[i]), '%.6f' % COST['params'][i], '%.5f' % COST['flops'][i],
                   '%.8f' % COST['latency'][i]]
            row += ['%.2f' % Y[d][i] for d in DS]
            row += ['%.4f' % pred[d][i] for d in DS]
            row += ['%.4f' % pred[s_][i] for t in DS for s_ in DS if s_ != t]
            wr.writerow(row)
    print('wrote', path, '(%d rows)' % len(keys))


def emit_table(o):
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering')
    A(r'\caption{The ranking signal on NAS-Bench-201~\cite{dong2020nasbench201}: '
      r'15{,}625 architectures, three datasets, a space we did not design. '
      r'Top: the same pairwise ridge, fitted on $N$ labels and ranking every architecture '
      r'held out, against the two zero-cost proxies the benchmark table supplies free. '
      r'Middle: fitted on 1{,}000 labels of one dataset and applied to another with '
      r'\emph{no target label}; \emph{own} is the same model fitted on the target. '
      r'Bottom: the free shortlist, as accuracy points lost against the best of the '
      r'14{,}625 architectures held out.}')
    A(r'\label{tab:nb201}')
    A(r'\footnotesize')
    A(r'\resizebox{\columnwidth}{!}{%')
    A(r'\begin{tabular}{lccc}')
    A(r'\toprule')
    A(r' & CIFAR-10 & CIFAR-100 & ImageNet16-120 \\')
    A(r'\midrule')
    A(r'\multicolumn{4}{l}{\emph{Rank fidelity $\rho$ (Kendall $\tau$) against final accuracy}} \\')
    for N in [100, 250, 1000]:
        c = [o['fidelity'][d][str(N)] for d in DS]
        A(r'\quad Genotype surrogate, %d labels & ' % N +
          ' & '.join('%.3f (%.3f)' % (x['rho'], x['tau']) for x in c) + r' \\')
    for pr, lab in [('params', 'Parameter count'), ('flops', 'FLOPs')]:
        c = [o['fidelity'][d][pr] for d in DS]
        A(r'\quad %s (zero-cost proxy) & ' % lab +
          ' & '.join('%.3f (%.3f)' % (x['rho'], x['tau']) for x in c) + r' \\')
    A(r'\addlinespace[2pt]\midrule')
    A(r'\multicolumn{4}{l}{\emph{Transfer with zero target labels: $\rho$ / own $\rho$}} \\')
    for src in DS:
        cells = []
        for tgt in DS:
            if tgt == src:
                cells.append('---')
            else:
                t = o['transfer'][f'{src}->{tgt}']
                cells.append('%.3f / %.3f' % (t['rho'], t['rho_fitted_here']))
        A(r'\quad fitted on %s & ' % SHORT[src] + ' & '.join(cells) + r' \\')
    A(r'\addlinespace[2pt]\midrule')
    A(r'\multicolumn{4}{l}{\emph{Regret of the free top-$k$ shortlist (accuracy points)}} \\')
    for k in [1, 5, 20]:
        c = [o['regret'][d][str(k)] for d in DS]
        A(r'\quad $k=%d$ & ' % k + ' & '.join('%.2f' % x['mean'] for x in c) + r' \\')
    A(r'\bottomrule')
    A(r'\end{tabular}}')
    A(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_nb201.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def emit_figure(o, P, Y):
    """Two figures: the transfer matrix alone for the main paper (single column), and the
    label-efficiency curve beside it for the supplement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'cifar10': '#1b6ca8', 'cifar100': '#c7522a', 'ImageNet16-120': '#2e7d4f'}
    NS = [100, 250, 500, 1000]

    M = np.full((3, 3), np.nan)
    for i, s_ in enumerate(DS):
        for j, t in enumerate(DS):
            if s_ != t:
                M[i, j] = o['transfer'][f'{s_}->{t}']['rho']

    def matrix(ax, fs=7.5):
        im = ax.imshow(M, cmap='YlGnBu', vmin=0.70, vmax=0.90)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, '%.3f' % M[i, j], ha='center', va='center', fontsize=fs,
                            color='white' if M[i, j] > 0.85 else 'black')
        ax.set_xticks(range(3), [SHORT[d] for d in DS], fontsize=fs - 0.5)
        ax.set_yticks(range(3), [SHORT[d] for d in DS], fontsize=fs - 0.5)
        ax.set_xlabel('applied to', fontsize=fs)
        ax.set_ylabel('fitted on', fontsize=fs)
        return im

    def curve(ax, fs=7.5):
        for d in DS:
            ax.plot(NS, [o['fidelity'][d][str(N)]['rho'] for N in NS], 'o-',
                    color=COL[d], label=SHORT[d], lw=1.5, ms=3.4)
            tr = [v['rho'] for k, v in o['transfer'].items() if k.endswith('->' + d)]
            ax.axhline(max(tr), color=COL[d], ls=':', lw=1.0)
        ax.set_xscale('log')
        ax.set_xlabel('labels used to fit the surrogate', fontsize=fs)
        ax.set_ylabel(r'Spearman $\rho$', fontsize=fs)
        ax.tick_params(labelsize=fs - 1)
        ax.legend(fontsize=fs - 1.5, frameon=False, loc='lower right', handlelength=1.4)
        ax.grid(alpha=.25, lw=.5)
        ax.text(0.03, 0.06, 'dotted: best transfer in,\nzero target labels',
                transform=ax.transAxes, fontsize=fs - 2, va='bottom')

    # main paper: the matrix alone
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    im = matrix(ax)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_nb201.' + ext),
                    dpi=220, bbox_inches='tight')
    plt.close(fig)

    # supplement: label efficiency beside it
    fig2, ax2 = plt.subplots(1, 2, figsize=(7.0, 2.7))
    curve(ax2[0], fs=8)
    im2 = matrix(ax2[1], fs=8)
    fig2.colorbar(im2, ax=ax2[1], fraction=.046, pad=.04).ax.tick_params(labelsize=6)
    fig2.tight_layout()
    for ext in ['pdf', 'png']:
        fig2.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_nb201_full.' + ext),
                     dpi=220, bbox_inches='tight')
    plt.close(fig2)
    print('wrote figures/fig_nb201.pdf and figures/fig_nb201_full.pdf')


if __name__ == '__main__':
    main()
