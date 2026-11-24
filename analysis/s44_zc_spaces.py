"""Does the free ranker survive in spaces whose genotype is nothing like ours?

`s35_nb201.py` tested the claim of Section~V on NAS-Bench-201, whose cell is six categorical
genes -- close enough to our own 13-gene space that a sceptic can call it a near-replicate.
This script runs the same model on the two remaining spaces NAS-Bench-Suite-Zero covers, both
of which encode an architecture very differently:

* **NAS-Bench-101**~\\cite{ying2019nasbench101} -- 9,781 architectures over a 7-node DAG. The
  genotype is a 50-bit adjacency pattern plus the operation of each intermediate node: mostly
  *binary* genes, where NAS-Bench-201 has none.
* **NAS-Bench-301**~\\cite{siems2020nasbench301} -- 11,221 architectures of the DARTS space.
  The genotype is two cells of eight (input, operation) choices: 32 genes of two interleaved
  kinds, and a space of $10^{18}$ rather than $10^{4}$.

Both are read from the files NAS-Bench-Suite-Zero releases, so the 13 proxies and the
accuracy of each architecture come from one source and no number here is ours to choose.
NAS-Bench-301's accuracies are a surrogate benchmark's predictions rather than measurements,
which is stated wherever the result is quoted.

The model is unchanged. Because these genotypes are long -- 56 genes gives 1,540 gene pairs --
the ridge is evaluated in its kernel form, which is exact and needs no design matrix:
for one-hot gene indicators plus all pairwise interaction indicators, the inner product
between two architectures that agree on $m$ genes is exactly $m + \\binom{m}{2}$. The
equivalence to the primal ridge of `surrogate.py` is asserted against `s35_nb201.py` on
NAS-Bench-201 before anything else runs.

Outputs
-------
results/zc_spaces.json       every number below
../tables/tab_zc_spaces.tex  Table: three spaces, the free ranker against the proxy family
"""
import ast, json, lzma, os, re, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau

import common
from s35_nb201 import regret
from s43_zc_proxies import (PROXIES, NICE, KS, NSEED, TOL, _open_maybe_xz, ranks,
                            kstar, zc_missing)

#: label budgets. 1,000 is the budget `s35_nb201.py` uses; the smaller ones say how fast the
#: model is identified in a space with many more genes.
NS = [100, 250, 500, 1000]

SPACES = {
    'nb101': {'file': 'zc_nasbench101.json', 'dataset': 'cifar10',
              'label': 'NAS-Bench-101', 'measured': True,
              'note': '9,781 of the 423,624 architectures, the subset the proxy file covers'},
    'nb301': {'file': 'zc_nasbench301.json', 'dataset': 'cifar10',
              'label': 'NAS-Bench-301', 'measured': False,
              'note': 'accuracies are the NAS-Bench-301 surrogate, not measurements'},
}


# --------------------------------------------------------------------------- the kernel ridge
def match_counts(Q, P):
    """Number of genes on which each row of Q agrees with each row of P."""
    out = np.zeros((len(Q), len(P)), dtype=np.int16)
    for g in range(Q.shape[1]):
        out += (Q[:, g][:, None] == P[None, :, g])
    return out


def kernel(Q, P):
    """Inner product of the one-hot-plus-pairwise design, in closed form.

    A gene indicator contributes 1 for each gene the two architectures share; a pairwise
    indicator contributes 1 for each *pair* of genes they both share, and if they agree on m
    genes there are exactly C(m,2) such pairs.
    """
    m = match_counts(Q, P).astype(np.float64)
    return m + m * (m - 1.0) / 2.0


def fit_kernel(P, y, alpha=10.0):
    """The ridge of `surrogate.py`, in its dual form. Returns a predict function."""
    y = np.asarray(y, float)
    b = float(y.mean())
    K = kernel(P, P)
    K[np.diag_indices_from(K)] += alpha
    a = np.linalg.solve(K, y - b)
    return lambda Q: kernel(Q, P) @ a + b


def check_equivalence():
    """The dual must reproduce the primal of s35_nb201.py exactly, or nothing below is the
    same model as the rest of the paper."""
    from s35_nb201 import load as load35, fit as fit_primal
    _keys, P, Y, _c = load35()
    idx = np.random.default_rng(0).permutation(len(P))
    tr, te = idx[:400], idx[400:1400]
    a = fit_primal(P[tr], Y['cifar10'][tr])(P[te])
    b = fit_kernel(P[tr], Y['cifar10'][tr])(P[te])
    d = float(np.abs(a - b).max())
    if d > 1e-6:
        sys.exit('the kernel form does not reproduce the primal ridge (max |diff| = %.3g)' % d)
    print('kernel form verified against the primal ridge (max |diff| = %.1e)' % d)
    return d


# --------------------------------------------------------------------------- genotypes
#: NAS-Bench-101 modules have between 2 and 7 nodes, so their keys are of five different
#: lengths. A fixed-length genotype is obtained by embedding each module in the largest one.
NB101_MAX_NODES = 7
#: the gene value that stands for "this node does not exist in this module".
ABSENT = 9


def _flatten(k):
    v = ast.literal_eval(k)
    out = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
        else:
            out.append(int(x))
    walk(v)
    return out


def genes_from_key(k, space):
    """Turn a NAS-Bench-Suite-Zero architecture key into a fixed-length categorical genotype.

    NAS-Bench-301 keys are already fixed-length: two cells of eight (input, operation) pairs,
    which flatten to 32 genes of two interleaved kinds.

    NAS-Bench-101 keys are a $V \\times V$ adjacency matrix followed by $V$ node labels, and
    $V$ runs from 2 to 7 across the 9,781 architectures (8,268 have the full seven). Each
    module is embedded in the seven-node one: the adjacency matrix goes into the top-left
    block of a $7\\times7$ matrix of zeros and the labels into the first $V$ slots, with the
    remaining nodes given a distinct `absent` label. A missing node is then a gene value like
    any other, no architecture is discarded, and the genes that end up constant across the
    space -- the structurally empty lower triangle among them -- are dropped before fitting.
    """
    f = _flatten(k)
    if space != 'nb101':
        return f
    # V^2 adjacency entries followed by V labels
    V = int(round((-1 + (1 + 4 * len(f)) ** 0.5) / 2))
    if V * V + V != len(f):
        sys.exit('NAS-Bench-101 key of unexpected length %d: %r' % (len(f), k))
    adj = np.array(f[:V * V], dtype=np.int16).reshape(V, V)
    lab = f[V * V:]
    M = np.zeros((NB101_MAX_NODES, NB101_MAX_NODES), dtype=np.int16)
    M[:V, :V] = adj
    out = list(M.reshape(-1))
    out += list(lab) + [ABSENT] * (NB101_MAX_NODES - V)
    return [int(x) for x in out]


def load_space(name):
    cfg = SPACES[name]
    path = None
    for cand in [cfg['file'] + '.xz', cfg['file']]:
        p = os.path.join(common.DATA, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        zc_missing(cfg['file'], name)
    with _open_maybe_xz(path) as fh:
        raw = json.load(fh)
    ds = cfg['dataset']
    if ds not in raw:
        sys.exit('%s has no %s' % (path, ds))
    keys = sorted(raw[ds])
    G = [genes_from_key(k, name) for k in keys]
    L = {len(g) for g in G}
    if len(L) != 1:
        sys.exit('%s: architecture keys of differing length %s' % (name, sorted(L)))
    P = np.array(G, dtype=np.int16)
    y = np.array([raw[ds][k]['val_accuracy'] for k in keys], float)
    S, T = {}, {}
    for m in PROXIES:
        S[m] = np.array([(raw[ds][k][m]['score'] if isinstance(raw[ds][k][m], dict)
                          else raw[ds][k][m]) for k in keys], float)
        T[m] = np.array([(raw[ds][k][m].get('time', np.nan)
                          if isinstance(raw[ds][k][m], dict) else np.nan) for k in keys], float)
    # drop genes that never vary: they carry no information and would only inflate the
    # feature count. Report how many were dropped.
    keep = np.array([len(np.unique(P[:, g])) > 1 for g in range(P.shape[1])])
    info = {'n_arch': len(keys), 'n_genes_raw': int(P.shape[1]),
            'n_genes': int(keep.sum()),
            'cardinalities': [int(len(np.unique(P[:, g]))) for g in np.where(keep)[0]],
            'label': cfg['label'], 'dataset': ds, 'measured': cfg['measured'],
            'note': cfg['note'], 'source': os.path.basename(path),
            'acc_mean': float(y.mean()), 'acc_max': float(y.max()),
            'acc_sd': float(y.std())}
    return P[:, keep], y, S, T, info


def fix_signs(S, y):
    sign = {}
    out = {}
    for m in PROXIES:
        v = S[m].astype(float)
        ok = np.isfinite(v)
        r = spearmanr(v[ok], y[ok]).correlation
        sign[m] = -1.0 if (r is not None and r < 0) else 1.0
        v = sign[m] * v
        bad = ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = np.nanmin(v[~bad]) - 1e6
        out[m] = v
    return out, sign


def main():
    out = {'equivalence_max_abs_diff': check_equivalence(),
           'n_seeds': NSEED, 'ks': KS, 'tol': TOL, 'label_budgets': NS}
    for name in SPACES:
        got = load_space(name)
        if got is None:
            continue
        P, y, S_raw, T, info = got
        S, sign = fix_signs(S_raw, y)
        n = len(y)
        print('%s: %d architectures, %d varying genes, cardinalities %s'
              % (name, n, info['n_genes'], sorted(set(info['cardinalities']))))

        whole = {m: float(spearmanr(S[m], y).correlation) for m in PROXIES}
        best = max(PROXIES, key=lambda m: whole[m])
        top3 = sorted(PROXIES, key=lambda m: -whole[m])[:3]

        A = {}
        for key in ([('surrogate', str(N)) for N in NS]
                    + [('proxy', m) for m in PROXIES]
                    + [('fused', 'all13'), ('fused', 'top3'),
                       ('fused', 'surrogate+best'), ('random', '')]):
            A[key] = {'rho': [], 'tau': [], 'rg': {k: [] for k in KS}}

        def add(pred, yt, key):
            e = A[key]
            e['rho'].append(spearmanr(pred, yt).correlation)
            e['tau'].append(kendalltau(pred, yt).correlation)
            for k in KS:
                e['rg'][k].append(regret(pred, yt, k))

        NMAX = max(NS)
        for sd in range(NSEED):
            g = np.random.default_rng(sd)
            idx = g.permutation(n)
            tr, te = idx[:NMAX], idx[NMAX:]
            yt = y[te]
            preds = {}
            for N in NS:
                p = fit_kernel(P[tr[:N]], y[tr[:N]])(P[te])
                preds[N] = p
                add(p, yt, ('surrogate', str(N)))
            rk = {}
            for m in PROXIES:
                add(S[m][te], yt, ('proxy', m))
                rk[m] = ranks(S[m][te])
            add(np.mean([rk[m] for m in PROXIES], axis=0), yt, ('fused', 'all13'))
            add(np.mean([rk[m] for m in top3], axis=0), yt, ('fused', 'top3'))
            add((2 * ranks(preds[NMAX]) + rk[best]) / 3.0, yt, ('fused', 'surrogate+best'))
            add(g.standard_normal(len(te)), yt, ('random', ''))

        res = {}
        for key, e in A.items():
            rg = {k: float(np.mean(v)) for k, v in e['rg'].items()}
            res['|'.join(key)] = {
                'rho': float(np.mean(e['rho'])), 'rho_sd': float(np.std(e['rho'])),
                'tau': float(np.mean(e['tau'])),
                'regret': {str(k): v for k, v in rg.items()},
                'k_star': kstar(rg)}

        # the paired contrast, on the shared splits
        def pair(a, b, k=10):
            x = np.array(A[a]['rg'][k]) - np.array(A[b]['rg'][k])
            lo, hi = np.percentile(
                [np.mean(x[np.random.default_rng(i).integers(0, len(x), len(x))])
                 for i in range(2000)], [2.5, 97.5])
            return {'mean': float(x.mean()), 'ci': [float(lo), float(hi)],
                    'wins': int((x < 0).sum()), 'n': int(len(x))}

        info.update(
            whole_table_rho=whole, best_proxy=best, top3_proxies=top3, signs=sign,
            n_held_out=n - NMAX,
            cost={m: {'sec_per_arch': (float(np.nanmean(T[m]))
                                       if np.isfinite(T[m]).any() else None)}
                  for m in PROXIES},
            rankers=res,
            paired={'surrogate_minus_bestproxy': pair(('surrogate', str(NMAX)),
                                                      ('proxy', best)),
                    'bestproxy_minus_random': pair(('proxy', best), ('random', '')),
                    'fused_minus_bestproxy': pair(('fused', 'surrogate+best'),
                                                  ('proxy', best))})
        out[name] = info

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'zc_spaces.json'), 'w'), indent=1)
    emit_table(out)
    print(json.dumps({
        k: {'n': out[k]['n_arch'], 'genes': out[k]['n_genes'],
            'surrogate_rho': round(out[k]['rankers']['surrogate|1000']['rho'], 3),
            'best_proxy': out[k]['best_proxy'],
            'best_proxy_rho': round(out[k]['rankers']['proxy|' + out[k]['best_proxy']]['rho'], 3),
            'surrogate_r10': round(out[k]['rankers']['surrogate|1000']['regret']['10'], 3),
            'proxy_r10': round(out[k]['rankers']['proxy|' + out[k]['best_proxy']]['regret']['10'], 3),
            'random_r10': round(out[k]['rankers']['random|']['regret']['10'], 3),
            'paired': {a: [round(b['mean'], 3), [round(c, 3) for c in b['ci']]]
                       for a, b in out[k]['paired'].items()}}
        for k in SPACES if k in out}, indent=1))


def emit_table(o):
    names = [k for k in SPACES if k in o]
    if not names:
        return
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering')
    A(r'\caption{The same free ranker in two further spaces it was not designed for, against '
      r'the zero-cost proxy family. Both spaces and all 13 proxy scores come from '
      r'NAS-Bench-Suite-Zero~\cite{krishnakumar2022nasbenchsuitezero}; the surrogate is '
      r'fitted on 1{,}000 labels and every ranker is scored on the same held-out pool over '
      r'20 splits. $r_{10}$ is the accuracy lost by training the ten best-ranked candidates '
      r'instead of the whole pool. NAS-Bench-101 genotypes are mostly binary and '
      r'NAS-Bench-301 has 28 varying genes in a $10^{18}$ space, so neither resembles the 13-gene '
      r'grid the model was built for. NAS-Bench-301 accuracies are that benchmark''s '
      r'surrogate predictions, not measurements.}')
    A(r'\label{tab:zc_spaces}')
    A(r'\footnotesize')
    A(r'\setlength{\tabcolsep}{4pt}')
    A(r'\begin{tabular}{l cc cc}')
    A(r'\toprule')
    A(r'& \multicolumn{2}{c}{NAS-Bench-101} & \multicolumn{2}{c}{NAS-Bench-301} \\')
    A(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    A(r'Ranker & $\rho$ & $r_{10}$ & $\rho$ & $r_{10}$ \\')
    A(r'\midrule')

    def row(label, key, bold=False):
        cs = []
        for nm in names:
            e = o[nm]['rankers'].get(key)
            cs += ['---', '---'] if e is None else ['%.3f' % e['rho'],
                                                    '%.2f' % e['regret']['10']]
        lab = r'\textbf{%s}' % label if bold else label
        A(r'\quad ' + lab + ' & ' + ' & '.join(cs) + r' \\')

    row('Genotype surrogate, 1{,}000 labels', 'surrogate|1000', bold=True)
    row('Genotype surrogate, 100 labels', 'surrogate|100')
    A(r'\addlinespace[1pt]')
    for m in ['synflow', 'nwot', 'jacov', 'params', 'flops']:
        row(NICE[m] + ' (zero-cost)', 'proxy|' + m)
    row('Rank-mean of all 13 proxies', 'fused|all13')
    A(r'\addlinespace[1pt]')
    row('Surrogate $+$ best proxy (2:1)', 'fused|surrogate+best')
    row('Random shortlist', 'random|')
    A(r'\midrule')
    A(r'Architectures / varying genes & '
      + ' & '.join(r'\multicolumn{2}{c}{%s / %d}' % ('{:,}'.format(o[nm]['n_arch']).replace(',', '{,}'),
                                                     o[nm]['n_genes']) for nm in names) + r' \\')
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_zc_spaces.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


if __name__ == '__main__':
    main()
