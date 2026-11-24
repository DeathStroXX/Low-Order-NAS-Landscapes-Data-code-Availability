"""Exact functional ANOVA of a complete factorial, for any shape.

`s53_spectrum.py` carries its own copy of this specialised to NAS-Bench-201's 5^6 grid and is
left untouched so that the numbers behind the original tables stay byte-identical. This module
is the general version the later scripts need, because the other benchmarks that satisfy the
exactness condition are not 5^6: NAS-Bench-Macro is 3^8 and NATS-Bench's size space is 8^5.

The decomposition is the same one `\\eqref{eq:anova}` states. For a function f observed once at
every cell of a product grid,

    f(x) = f_0 + sum_a f_a(x_a) + sum_{a<b} f_ab(x_a,x_b) + ... + f_{1..d}(x),

the components are orthogonal, have mean zero on every axis they depend on, and are given by
Moebius inversion over marginal means. Nothing is fitted and there is no estimation error, so
the variance each order carries is a property of the tabulated function and not of a model of
it. Two implementations are provided and cross-checked against each other:

  * `components` enumerates all 2^d subsets. It returns every component separately, which is
    what the per-edge order-one analysis needs, and it is the one `s53` uses. Cost is
    O(2^d * N), fine to d = 10 or so.
  * `order_variance` is the projection recursion. Along each axis apply either the mean
    operator P (the factor does not appear in the component) or Q = I - P (it does), and carry
    one running tensor per order: T_k <- P T_k + Q T_{k-1}. Cost is O(d^2 * N) in time and
    O(d * N) in memory, so the order spectrum stays computable for grids where enumerating
    subsets would not be.

Both are exact; `check_consistency` asserts they agree, and that the shares sum to one, which
is the orthogonality of the decomposition restated as a test.
"""
import itertools
import math

import numpy as np


def to_grid(G, y, shape):
    """Scatter observations into the factorial grid, verifying the design is complete.

    A complete factorial with exactly one observation per cell is the whole exactness
    condition, so it is checked here rather than assumed anywhere downstream.
    """
    G = np.asarray(G)
    y = np.asarray(y, dtype=float)
    if G.ndim != 2 or G.shape[0] != y.shape[0]:
        raise ValueError('G must be (n, d) and y must be (n,)')
    if G.shape[1] != len(shape):
        raise ValueError('G has %d factors, shape has %d' % (G.shape[1], len(shape)))
    n_cells = int(np.prod(shape))
    if y.shape[0] != n_cells:
        raise ValueError('design is not complete: %d rows for %d cells' % (y.shape[0], n_cells))
    seen = np.zeros(shape, dtype=bool)
    idx = tuple(G.T)
    seen[idx] = True
    if not seen.all():
        raise ValueError('design is not complete: %d cells unobserved' % int((~seen).sum()))
    Y = np.empty(shape, dtype=float)
    Y[idx] = y
    return Y


def components(Y):
    """Every ANOVA component of a complete factorial, keyed by the subset of factors."""
    d = Y.ndim
    comp = {}
    for k in range(d + 1):
        for S in itertools.combinations(range(d), k):
            axes = tuple(a for a in range(d) if a not in S)
            m = Y.mean(axis=axes, keepdims=True) if axes else Y.copy()
            for j in range(k):
                for T in itertools.combinations(S, j):
                    m = m - comp[T]
            comp[S] = m
    return comp


def order_variance(Y):
    """Variance carried by each interaction order, by the projection recursion.

    Returns an array v of length d+1, where v[k] is the variance at order k and v[0] is zero
    (the grand mean carries no variance). Sum(v) is the variance of Y.
    """
    d = Y.ndim
    cur = [Y.astype(float, copy=True)] + [None] * d
    for axis in range(d):
        nxt = [None] * (d + 1)
        for k in range(min(axis + 1, d) + 1):
            if cur[k] is None:
                continue
            m = cur[k].mean(axis=axis, keepdims=True)
            # P T_k : the factor is averaged out, the order does not rise
            nxt[k] = m if nxt[k] is None else nxt[k] + m
            # Q T_k : the factor enters this component, the order rises by one
            q = cur[k] - m
            nxt[k + 1] = q if nxt[k + 1] is None else nxt[k + 1] + q
        cur = nxt
    v = np.zeros(d + 1)
    for k in range(d + 1):
        if cur[k] is not None:
            v[k] = float((np.broadcast_to(cur[k], Y.shape) ** 2).mean())
    v[0] = 0.0
    return v


def order_sums(Y):
    """The sum of all components of each order, as full-grid arrays.

    `order_sums(Y)[d]` summed over d' <= d is the best degree-d approximation of Y in squared
    error, which is what the ceiling of Section IV is a statement about.
    """
    d = Y.ndim
    cur = [Y.astype(float, copy=True)] + [None] * d
    for axis in range(d):
        nxt = [None] * (d + 1)
        for k in range(min(axis + 1, d) + 1):
            if cur[k] is None:
                continue
            m = cur[k].mean(axis=axis, keepdims=True)
            nxt[k] = m if nxt[k] is None else nxt[k] + m
            q = cur[k] - m
            nxt[k + 1] = q if nxt[k + 1] is None else nxt[k + 1] + q
        cur = nxt
    out = {}
    for k in range(d + 1):
        out[k] = (np.broadcast_to(cur[k], Y.shape).copy() if cur[k] is not None
                  else np.zeros(Y.shape))
    return out


def spectrum(Y):
    """Share of variance by order, in percent, as a list indexed 1..d."""
    v = order_variance(Y)
    tot = v.sum()
    if tot <= 0:
        raise ValueError('the function is constant: no variance to decompose')
    return [100.0 * v[k] / tot for k in range(1, Y.ndim + 1)]


def null_spectrum(shape):
    """Where white noise puts its variance: prod(|g_a|-1 choose the chosen axes) per order.

    A function of independent noise has its variance spread uniformly over the
    prod(|g_a|) - 1 non-constant basis directions of the grid. The number of them at order k is
    the sum over k-subsets S of prod_{a in S} (|g_a| - 1), so for a uniform grid of cardinality
    q in d factors it is C(d,k)(q-1)^k. This is the control the concentration claim needs and
    it requires no data at all.
    """
    d = len(shape)
    n = []
    for k in range(1, d + 1):
        s = 0
        for S in itertools.combinations(range(d), k):
            p = 1
            for a in S:
                p *= (shape[a] - 1)
            s += p
        n.append(s)
    tot = float(sum(n))
    return [100.0 * x / tot for x in n]


def check_consistency(Y, atol=1e-8):
    """Assert the two implementations agree and that the shares sum to 100 percent."""
    v = order_variance(Y)
    s = sum(v)
    assert abs(s - Y.var()) < atol * max(1.0, abs(Y.var())), (s, Y.var())
    if Y.ndim <= 8:
        comp = components(Y)
        d = Y.ndim
        w = np.zeros(d + 1)
        for S, c in comp.items():
            if S:
                w[len(S)] += float((np.broadcast_to(c, Y.shape) ** 2).mean())
        assert np.allclose(v, w, atol=atol * max(1.0, abs(Y.var()))), (v, w)
    return True


def cumulative(sp):
    """Cumulative share of variance at order <= d, in percent."""
    out, run = [], 0.0
    for x in sp:
        run += x
        out.append(run)
    return out


def comb(n, k):
    return math.comb(n, k)
