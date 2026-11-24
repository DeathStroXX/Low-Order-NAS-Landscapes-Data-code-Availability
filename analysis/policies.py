"""Search policies on the surrogate benchmark (fractalbench.py), beyond the as-run GA.

`run` is the vectorized GA of fractalbench with the extra knobs the policy experiments need:
a warm-started initial population, per-gene mutation weights, a gene permutation for one-point
crossover, uniform crossover, frozen genes, and a noise level on the selection signal.
`screen_then_measure` and `random_measure` are the two policies that replace search altogether.
The model-in-the-loop policy lives in s31_sitl.py, where the model is genuinely refitted on
what has been measured.
"""
import numpy as np
from surrogate import CARD, NG


def _tournament(rng, P, f, n, k, R, pop):
    c = rng.integers(0, pop, (R, n, k))
    fv = np.take_along_axis(f[:, None, :].repeat(n, 1), c, 2)
    w = np.take_along_axis(c, fv.argmax(2)[:, :, None], 2)[:, :, 0]
    return np.take_along_axis(P, w[:, :, None].repeat(NG, 2), 1)


def run(bench, R=400, pop=20, gens=26, elite=2, k=3, pm=0.2, seed=1, order=None,
        uniform=False, gene_w=None, freeze=None, init=None, noise=0.0):
    """R replicates of a (gens x pop) GA. Returns per-replicate best / top-10 / mean."""
    rng = np.random.default_rng(seed)
    order = np.arange(NG) if order is None else np.asarray(order)
    inv = np.argsort(order)
    fz = np.array([freeze.get(g, -1) if freeze else -1 for g in range(NG)]) \
        if isinstance(freeze, dict) else np.full(NG, -1)
    P = rng.integers(0, CARD, (R, pop, NG))
    if init is not None:
        m = min(len(init), pop)
        P[:, :m] = np.asarray(init)[:m]
    for gi in np.where(fz >= 0)[0]:
        P[..., gi] = fz[gi]
    pmg = np.full(NG, pm) if gene_w is None else np.clip(pm * np.asarray(gene_w, float), 0, 1)
    best = np.full(R, -np.inf)
    sampled = []
    for g in range(gens):
        t = bench.score(P.reshape(-1, NG)).reshape(R, pop)
        f = t + rng.normal(0, noise, t.shape) if noise else t
        best = np.maximum(best, t.max(1))
        sampled.append(t)
        if g == gens - 1:
            break
        o = np.argsort(-f, 1)
        new = np.empty_like(P)
        new[:, :elite] = np.take_along_axis(P, o[:, :elite, None].repeat(NG, 2), 1)
        n = pop - elite
        A = _tournament(rng, P, f, n, k, R, pop)
        B = _tournament(rng, P, f, n, k, R, pop)
        if uniform:
            ch = np.where(rng.random((R, n, NG)) < 0.5, A, B)
        else:
            pt = rng.integers(1, NG, (R, n, 1))
            ch = np.where(np.arange(NG)[None, None, :] < pt,
                          A[:, :, order], B[:, :, order])[:, :, inv]
        mm = rng.random((R, n, NG)) < pmg[None, None, :]
        sh = rng.integers(1, np.broadcast_to(CARD, (R, n, NG)))
        new[:, elite:] = np.where(mm, (ch + sh) % CARD, ch)
        P = new
        for gi in np.where(fz >= 0)[0]:
            P[..., gi] = fz[gi]
    S = np.concatenate(sampled, 1)
    return dict(best=best, top10=np.sort(S, 1)[:, -10:].mean(1), mean=S.mean(1),
                n_measured=gens * pop)


def screen_then_measure(bench, rng, n_screen, n_measure, chunk=50000):
    """The policy that replaces search entirely: screen `n_screen` free, measure the best.

    Screening is chunked: the surrogate's design matrix is 1,572 columns wide, so a
    half-million-candidate screen is scored in blocks and only the running best kept.
    """
    keep_P, keep_s = None, None
    left = n_screen
    while left > 0:
        n = min(chunk, left)
        left -= n
        c = rng.integers(0, CARD, (n, NG))
        v = bench.score(c, tabular=False)
        if keep_P is None:
            keep_P, keep_s = c, v
        else:
            keep_P = np.concatenate([keep_P, c]); keep_s = np.concatenate([keep_s, v])
        if len(keep_s) > 4 * n_measure:
            o = np.argsort(-keep_s)[:n_measure]
            keep_P, keep_s = keep_P[o], keep_s[o]
    sel = keep_P[np.argsort(-keep_s)[:n_measure]]
    v = bench.score(sel)
    return float(v.max()), float(np.sort(v)[-10:].mean())


def random_measure(bench, rng, n_measure):
    """The floor: measure `n_measure` architectures drawn uniformly at random."""
    v = bench.score(rng.integers(0, CARD, (n_measure, NG)))
    return float(v.max()), float(np.sort(v)[-10:].mean())
