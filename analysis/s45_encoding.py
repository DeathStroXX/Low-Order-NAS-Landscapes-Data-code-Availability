"""Where the free ranker fails, why, and how a practitioner can tell in advance.

`s44_zc_spaces.py` produces one clear negative: on NAS-Bench-101 the genotype surrogate
reaches only $\\rho \\approx 0.50$ and is beaten by a zero-cost proxy, while on NAS-Bench-301
-- a space a thousand times larger -- the same model reaches $0.83$ and wins by a wide
margin. A result that holds in three spaces and fails in a fourth is worth more than a result
that is only ever reported where it works, provided the failure is explained and can be
anticipated. This script does both.

**The diagnosis.** NAS-Bench-101's genotype is a $7\\times7$ adjacency matrix: an edge is a
gene, and what a cell computes depends on the *paths* the edges form, not on the edges one or
two at a time. Two different adjacency matrices can also describe the same graph. Gene and
gene-pair indicators therefore cannot express the function, and no number of labels repairs
that -- which is exactly what the data show, and what the encoding literature
reports~\\cite{white2020encodings}. The other three spaces (ours, NAS-Bench-201,
NAS-Bench-301) are slot-structured: each gene names an independent design choice.

**The test.** If the encoding is the limit, then re-encoding the same architectures by the
paths from input to output -- still free, still a ridge, no training of anything -- should
recover the signal, and it does.

**The diagnostic a practitioner can run.** Fidelity against label count. Where the model is
merely under-identified, $\\rho$ climbs with labels; where the encoding is wrong, $\\rho$ is
flat. The gap $\\rho(1000) - \\rho(250)$ costs 750 extra labels to measure and tells the
laboratory which of the two situations it is in before it commits a budget.

Outputs
-------
results/encoding.json          every number below
../tables/tab_encoding.tex     Table: the diagnostic and the remedy
"""
import ast, json, lzma, os, sys
import numpy as np
from scipy.stats import spearmanr

import common
from s35_nb201 import regret
from s43_zc_proxies import KS, NSEED, ranks, kstar, zc_missing
from s44_zc_spaces import (NS, SPACES, fit_kernel, genes_from_key, load_space,
                           _flatten, _open_maybe_xz, fix_signs)

#: NAS-Bench-101 node labels, as the proxy file codes them (verified over all 9,781 keys:
#: node 0 is always 0 and the last node always 1).
NB101_INPUT, NB101_OUTPUT = 0, 1
NB101_OPS = (2, 3, 4)
#: paths are truncated at this many operations, as in the encoding literature.
MAX_PATH = 5
ALPHA = 10.0


def nb101_paths(k):
    """The set of input-to-output paths of one NAS-Bench-101 module, as operation sequences.

    The adjacency matrix is strictly upper triangular -- the benchmark stores its nodes in
    topological order -- so a depth-first walk from the input node terminates. A path is
    recorded as the tuple of operation labels of the intermediate nodes it passes through,
    which is what makes the feature invariant to how the nodes happen to be numbered.
    """
    f = _flatten(k)
    V = int(round((-1 + (1 + 4 * len(f)) ** 0.5) / 2))
    adj = np.array(f[:V * V], dtype=np.int8).reshape(V, V)
    lab = f[V * V:]
    out = set()

    def walk(node, ops):
        if node == V - 1:
            out.add(tuple(ops))
            return
        if len(ops) >= MAX_PATH:
            return
        for nxt in range(node + 1, V):
            if adj[node, nxt]:
                walk(nxt, ops + ([lab[nxt]] if nxt != V - 1 else []))
    walk(0, [])
    return out


def path_design(keys):
    """Binary design matrix over the paths that actually occur in the space."""
    sets = [nb101_paths(k) for k in keys]
    vocab = sorted(set().union(*sets))
    ix = {p: i for i, p in enumerate(vocab)}
    A = np.zeros((len(keys), len(vocab)), dtype=np.float64)
    for r, s in enumerate(sets):
        for p in s:
            A[r, ix[p]] = 1.0
    return A, vocab


def fit_linear(A, y, alpha=ALPHA):
    """Ridge on a given design matrix, the same penalty and intercept rule as everywhere."""
    b = float(np.mean(y))
    G = A.T @ A
    G[np.diag_indices_from(G)] += alpha
    w = np.linalg.solve(G, A.T @ (np.asarray(y, float) - b))
    return lambda B: B @ w + b


def summarise(pred, y):
    rg = {k: regret(pred, y, k) for k in KS}
    return {'rho': spearmanr(pred, y).correlation, 'rg': rg}


def main():
    out = {'n_seeds': NSEED, 'max_path_ops': MAX_PATH, 'label_budgets': NS,
           'alpha': ALPHA}

    # ----------------------------------------------------------------- the remedy, on nb101
    path = None
    for cand in ['zc_nasbench101.json.xz', 'zc_nasbench101.json']:
        p = os.path.join(common.DATA, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        zc_missing('zc_nasbench101.json', 'nb101')
    with _open_maybe_xz(path) as fh:
        raw = json.load(fh)['cifar10']
    keys = sorted(raw)
    y = np.array([raw[k]['val_accuracy'] for k in keys], float)
    P = np.array([genes_from_key(k, 'nb101') for k in keys], dtype=np.int16)
    keep = np.array([len(np.unique(P[:, g])) > 1 for g in range(P.shape[1])])
    P = P[:, keep]
    A, vocab = path_design(keys)
    n = len(keys)
    NMAX = max(NS)
    print('nb101: %d architectures, %d varying genes, %d distinct paths'
          % (n, P.shape[1], len(vocab)))

    acc = {('genotype', str(N)): [] for N in NS}
    acc.update({('path', str(N)): [] for N in NS})
    for sd in range(NSEED):
        idx = np.random.default_rng(sd).permutation(n)
        tr, te = idx[:NMAX], idx[NMAX:]
        for N in NS:
            s = tr[:N]
            acc[('genotype', str(N))].append(
                summarise(fit_kernel(P[s], y[s])(P[te]), y[te]))
            acc[('path', str(N))].append(
                summarise(fit_linear(A[s], y[s])(A[te]), y[te]))

    def collapse(rows):
        rg = {k: float(np.mean([r['rg'][k] for r in rows])) for k in KS}
        return {'rho': float(np.mean([r['rho'] for r in rows])),
                'rho_sd': float(np.std([r['rho'] for r in rows])),
                'regret': {str(k): v for k, v in rg.items()},
                'k_star': kstar(rg)}

    out['nb101'] = {
        'n_arch': n, 'n_genes': int(P.shape[1]), 'n_paths': len(vocab),
        'rankers': {'|'.join(k): collapse(v) for k, v in acc.items()},
        'paired_path_minus_genotype_regret10': _pair(
            [r['rg'][10] for r in acc[('path', str(NMAX))]],
            [r['rg'][10] for r in acc[('genotype', str(NMAX))]])}

    # ----------------------------------------------------------------- the diagnostic
    # In-sample against held-out R^2, at the same 1,000 labels, in every space the paper has.
    # A misspecified encoding is worse than the mean out of sample; an under-identified one is
    # not. Nothing here costs a label that was not already spent.
    diag = {'nb101_genotype': _r2(P, y, 'kernel'),
            'nb101_path': _r2(A, y, 'linear')}
    from s35_nb201 import load as load35, fit as fit_primal
    _k, P201, Y201, _c = load35()
    for d in Y201:
        diag['nb201_%s_genotype' % d] = _r2(P201, Y201[d], 'primal')
    got = load_space('nb301')
    if got is not None:
        P3, y3, _s, _t, _i = got
        diag['nb301_genotype'] = _r2(P3, y3, 'kernel')
    out['diagnostic'] = diag

    # the rejected diagnostic, recorded so the reader does not spend the labels
    rej = {'nb101_genotype': _slope(out['nb101']['rankers'], 'genotype'),
           'nb101_path': _slope(out['nb101']['rankers'], 'path')}
    zc = os.path.join(common.RES, 'zc_spaces.json')
    if os.path.exists(zc):
        sp = json.load(open(zc))
        if 'nb301' in sp:
            rej['nb301_genotype'] = _slope(sp['nb301']['rankers'], 'surrogate')
    nb = os.path.join(common.RES, 'nb201.json')
    if os.path.exists(nb):
        f = json.load(open(nb))['fidelity']
        for d in f:
            rej['nb201_%s_genotype' % d] = {
                'rho_250': f[d]['250']['rho'], 'rho_1000': f[d]['1000']['rho'],
                'gain': f[d]['1000']['rho'] - f[d]['250']['rho']}
    out['rejected_diagnostic_label_gain'] = rej

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'encoding.json'), 'w'), indent=1)
    emit_table(out)
    print(json.dumps({
        'nb101_genotype_rho': {N: round(out['nb101']['rankers']['genotype|%d' % N]['rho'], 3)
                               for N in NS},
        'nb101_path_rho': {N: round(out['nb101']['rankers']['path|%d' % N]['rho'], 3)
                           for N in NS},
        'nb101_genotype_r10': round(out['nb101']['rankers']['genotype|1000']['regret']['10'], 3),
        'nb101_path_r10': round(out['nb101']['rankers']['path|1000']['regret']['10'], 3),
        'nb101_path_kstar': out['nb101']['rankers']['path|1000']['k_star'],
        'diagnostic_r2': {k: [round(v['train'], 3), round(v['test'], 3)]
                          for k, v in diag.items()},
        'rejected_label_gain': {k: round(v['gain'], 3) for k, v in rej.items()}},
        indent=1))


def _r2(X, y, kind, N=1000, seeds=NSEED):
    """In-sample and held-out R^2 of the ridge, on the same splits used everywhere."""
    from s35_nb201 import fit as fit_primal
    n = len(y)
    tr_r2, te_r2 = [], []
    for sd in range(seeds):
        idx = np.random.default_rng(sd).permutation(n)
        tr, te = idx[:N], idx[N:]
        if kind == 'kernel':
            f = fit_kernel(X[tr], y[tr])
            ptr, pte = f(X[tr]), f(X[te])
        elif kind == 'primal':
            f = fit_primal(X[tr], y[tr])
            ptr, pte = f(X[tr]), f(X[te])
        else:
            f = fit_linear(X[tr], y[tr])
            ptr, pte = f(X[tr]), f(X[te])
        tr_r2.append(_rsq(ptr, y[tr]))
        te_r2.append(_rsq(pte, y[te]))
    return {'train': float(np.mean(tr_r2)), 'test': float(np.mean(te_r2)),
            'gap': float(np.mean(tr_r2) - np.mean(te_r2)), 'n_fit': N}


def _rsq(p, y):
    y = np.asarray(y, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _slope(rankers, prefix):
    a = rankers['%s|250' % prefix]['rho']
    b = rankers['%s|1000' % prefix]['rho']
    return {'rho_250': a, 'rho_1000': b, 'gain': b - a}


def _pair(a, b):
    x = np.array(a) - np.array(b)
    lo, hi = np.percentile(
        [np.mean(x[np.random.default_rng(i).integers(0, len(x), len(x))])
         for i in range(2000)], [2.5, 97.5])
    return {'mean': float(x.mean()), 'ci': [float(lo), float(hi)],
            'wins': int((x < 0).sum()), 'n': int(len(x))}


def emit_table(o):
    e = o['nb101']['rankers']
    d = o['diagnostic']
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering')
    A(r'\caption{The one space where the free ranker fails, and the diagnostic that '
      r'anticipates it. Top: on NAS-Bench-101 the genotype encoding is an adjacency matrix, '
      r'so what a cell computes depends on paths rather than on single edges; the ridge '
      r'saturates and more labels do not help. Re-encoding the same architectures by their '
      r'input-to-output paths---still a ridge, still no training---recovers the signal. '
      r'Bottom: held-out $R^{2}$, from the same 1{,}000 labels, separates a model that is '
      r'merely under-identified from one whose features cannot express the function at all, '
      r'and costs nothing. A ranker with $R^{2} \le 0$ out of sample is worse than no '
      r'ranker. The fidelity gain from 250 to 1{,}000 labels was tried as the diagnostic '
      r'first and does not separate the cases; it is in \texttt{results/encoding.json}.}')
    A(r'\label{tab:encoding}')
    A(r'\footnotesize')
    A(r'\setlength{\tabcolsep}{4pt}')
    A(r'\begin{tabular}{lcccc}')
    A(r'\toprule')
    A(r'\multicolumn{5}{l}{\emph{NAS-Bench-101, %s architectures, %d held out}} \\'
      % ('{:,}'.format(o['nb101']['n_arch']).replace(',', '{,}'),
         o['nb101']['n_arch'] - max(NS)))
    A(r'Encoding & $\rho$, 100 & $\rho$, 1{,}000 & $r_{10}$ & $k^{\star}$ \\')
    A(r'\midrule')
    for lab, pre in [('Genes and gene pairs (Sec.~III)', 'genotype'),
                     ('Input-to-output paths ($\\leq$%d ops)' % o['max_path_ops'], 'path')]:
        k = e['%s|1000' % pre]['k_star']
        A(r'\quad %s & %.3f & %.3f & %.2f & %s \\'
          % (lab, e['%s|100' % pre]['rho'], e['%s|1000' % pre]['rho'],
             e['%s|1000' % pre]['regret']['10'], '$>$500' if k is None else str(k)))
    A(r'\addlinespace[2pt]\midrule')
    A(r'\multicolumn{5}{l}{\emph{The diagnostic, from the same 1{,}000 labels and no more}} \\')
    A(r'Space and encoding & $R^{2}$ in & $R^{2}$ out & gap & verdict \\')
    A(r'\midrule')
    NAME = {'nb101_genotype': 'NAS-Bench-101, genes',
            'nb101_path': 'NAS-Bench-101, paths',
            'nb301_genotype': 'NAS-Bench-301, genes',
            'nb201_cifar10_genotype': 'NAS-Bench-201 (C10), genes',
            'nb201_cifar100_genotype': 'NAS-Bench-201 (C100), genes',
            'nb201_ImageNet16-120_genotype': 'NAS-Bench-201 (IN16), genes'}
    for key in ['nb101_genotype', 'nb101_path', 'nb201_cifar10_genotype',
                'nb201_cifar100_genotype', 'nb201_ImageNet16-120_genotype',
                'nb301_genotype']:
        if key not in d:
            continue
        v = d[key]
        verdict = r'\textbf{do not rank}' if v['test'] <= 0 else 'usable'
        A(r'\quad %s & %.3f & %+.3f & %.3f & %s \\'
          % (NAME[key], v['train'], v['test'], v['gap'], verdict))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_encoding.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


if __name__ == '__main__':
    main()
