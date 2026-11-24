"""The zero-cost genotype surrogate used throughout the new analyses.

A ridge regression on the 13 genes' one-hot indicators plus every gene-pair interaction
(1,572 features). It costs no training epoch and no LLM call, it is fitted in under a second,
and prediction is one matrix product, which is what lets the surrogate-benchmark harness run a
thousand GA replicates. On held-out architectures it ranks better than a full training epoch
(see s16_surrogate.py).

Genotypes are passed as integer code matrices (n, 13) in the gene order of `GENES`, with the
value order of `SS`; `Encoder.codes_of` converts a chromosome dict.
"""
import itertools, math
import numpy as np

SS = {'n_columns': [2, 3, 4], 'n_blocks': [2, 3, 4], 'base_channels': [16, 32, 64],
      'dropout_prob': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
      'fc_dropout': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
      'lr': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01],
      'momentum': [0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98],
      'activation': ['ReLU', 'GELU', 'LeakyReLU', 'SiLU'], 'kernel_size': [1, 3, 5, 7],
      'pooling_type': ['Max', 'Avg'], 'conv_type': ['Standard', 'Depthwise'],
      'norm_type': ['BatchNorm', 'InstanceNorm'], 'optimizer_type': ['SGD', 'AdamW', 'RMSprop']}
GENES = list(SS)
CARD = np.array([len(SS[g]) for g in GENES])
NG = len(GENES)
OFF = np.concatenate([[0], np.cumsum(CARD)[:-1]])
NMAIN = int(CARD.sum())
PAIRS = list(itertools.combinations(range(NG), 2))
_poff, _p = [], NMAIN
for a, b in PAIRS:
    _poff.append(_p)
    _p += CARD[a] * CARD[b]
PAIR_OFF = np.array(_poff)
NFEAT = _p
GRID_SIZE = int(np.prod(CARD.astype(object)))


def code(g, v):
    for i, x in enumerate(SS[g]):
        if isinstance(x, str):
            if str(v) == x:
                return i
        else:
            try:
                if math.isclose(float(v), float(x), rel_tol=1e-9, abs_tol=1e-12):
                    return i
            except (TypeError, ValueError):
                pass
    return 0


def codes_of(chrom):
    return np.array([code(g, chrom[g]) for g in GENES], dtype=np.int16)


def chrom_of(c):
    return {g: SS[g][int(v)] for g, v in zip(GENES, c)}


def design(P):
    """(n, 13) codes -> (n, NFEAT) sparse-as-dense design matrix (main + all pairs)."""
    P = np.asarray(P, int)
    n = len(P)
    A = np.zeros((n, NFEAT), dtype=np.float32)
    r = np.arange(n)
    A[r[:, None], (OFF[None, :] + P)] = 1.0
    for pi, (a, b) in enumerate(PAIRS):
        A[r, PAIR_OFF[pi] + P[:, a] * CARD[b] + P[:, b]] = 1.0
    return A


class PairwiseRidge:
    """Ridge on main effects + all gene-pair interactions, solved in the normal equations."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0

    def fit(self, P, y):
        A = design(P)
        y = np.asarray(y, float)
        self.b = float(y.mean())
        G = A.T @ A
        G[np.diag_indices_from(G)] += self.alpha
        self.w = np.linalg.solve(G, A.T @ (y - self.b))
        return self

    def predict(self, P):
        return design(P) @ self.w + self.b

    @staticmethod
    def select_alpha(P, y, alphas=(0.3, 1, 3, 10, 30, 100, 300), folds=5, seed=0):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        best, bestа = None, None
        for a in alphas:
            p = np.zeros(len(y))
            for f in range(folds):
                te = idx[f::folds]
                tr = np.setdiff1d(idx, te)
                p[te] = PairwiseRidge(a).fit(P[tr], y[tr]).predict(P[te])
            r = spearmanr(p, y).statistic
            if best is None or r > best:
                best, bestа = r, a
        return bestа, best


class GramFitter:
    """Fast repeated ridge fits over subsets of one fixed design matrix.

    Caches A = design(P) and the Gram matrix A^T A once; a fit that leaves rows out is then
    G_sub = G_full - A_out^T A_out, which makes leave-one-run-out over all 34 runs and the
    learning-curve sweep cheap.
    """

    def __init__(self, P, y, alpha=10.0):
        self.A = design(P).astype(np.float64)
        self.y = np.asarray(y, float)
        self.alpha = alpha
        self.G = self.A.T @ self.A
        self.c = self.A.T @ self.y
        self.s = self.A.sum(0)
        self.n = len(y)
        self.sy = self.y.sum()

    def fit_mask(self, keep):
        """keep: boolean mask of training rows. Returns predict(index_array)."""
        A, y = self.A, self.y
        if keep.all():
            G, c, s, n, sy = self.G, self.c, self.s, self.n, self.sy
        else:
            Ao = A[~keep]
            G = self.G - Ao.T @ Ao
            c = self.c - Ao.T @ y[~keep]
            s = self.s - Ao.sum(0)
            n = int(keep.sum())
            sy = float(y[keep].sum())
        b = sy / n
        Gr = G.copy()
        Gr[np.diag_indices_from(Gr)] += self.alpha
        w = np.linalg.solve(Gr, c - b * s)
        return lambda idx: A[idx] @ w + b
