"""Statistical helpers shared by the s6-s12 scripts. Conventions follow s1_statistics.py:
exact two-sided permutation tests on the difference of run means, Hedges' g with the
small-sample correction, Holm-Bonferroni over a named family, 95% t-intervals over runs."""
import itertools
import numpy as np
from scipy import stats


def perm_p(a, b):
    """Exact two-sided permutation test on the difference of means (all relabelings)."""
    allv = np.asarray(list(a) + list(b), float); n = len(a)
    obs = np.mean(a) - np.mean(b); cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool); m[list(idx)] = True
        tot += 1; cnt += abs(allv[m].mean() - allv[~m].mean()) >= abs(obs) - 1e-12
    return cnt / tot, tot


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1]); m = len(items); out = {}; prev = 0.0
    for i, (k, p) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * p)); out[k] = prev
    return out


def separated(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.min() > b.max() or a.max() < b.min())


def contrast(a, b):
    """Full report for one contrast between two sets of run-level values."""
    p, n = perm_p(a, b)
    return dict(d=float(np.mean(a) - np.mean(b)), g=hedges_g(a, b), p=float(p), n_perm=int(n),
                sep=separated(a, b), mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                sd_a=float(np.std(a, ddof=1)), sd_b=float(np.std(b, ddof=1)),
                n_a=len(a), n_b=len(b))


def brown_forsythe(a, b):
    return float(stats.levene(np.asarray(a, float), np.asarray(b, float), center='median').pvalue)


def norm_entropy(values):
    """Entropy of an empirical distribution over a gene's observed grid, normalized to [0,1]."""
    v = [x for x in values if x is not None and x == x]
    if not v:
        return float('nan')
    _, c = np.unique(np.asarray(v, dtype=object).astype(str), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
