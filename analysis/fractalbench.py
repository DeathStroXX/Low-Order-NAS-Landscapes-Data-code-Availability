"""A surrogate benchmark over the 13-gene FractalNet space.

The 9,508 architectures evaluated by the 34 released search runs are enough to fit a
zero-cost surrogate over the whole 42.5-million-point grid: a ridge on gene indicators plus
all gene-pair interactions (surrogate.py), which ranks held-out architectures better than a
full training epoch does (s16_surrogate.py). Querying it is one matrix product instead of
three training epochs, which turns every ablation the original study could not afford --
segment length, selection pressure, mutation rate, and the 90 LLM-written operators -- into
a sweep with a thousand replicates.

Two query modes, as in surrogate NAS benchmarks generally:
  tabular   -- the measured 3-epoch accuracy, for a genotype that was actually evaluated;
  surrogate -- the model's estimate everywhere else.
`Bench.score` prefers the tabular value and falls back to the surrogate.

The GA harness runs R independent replicates in lockstep: operators are called per individual
(ordinary Python functions, including the LLM-written ones) while all fitness queries of a
generation are served in one batched call.
"""
import os, random, math, textwrap, signal
import numpy as np
import common
import surrogate as sg
from surrogate import SS, GENES, CARD, NG, GRID_SIZE, codes_of, chrom_of, code


class Bench:
    """Fitness oracle for one dataset."""

    def __init__(self, ds, alpha=10.0):
        self.ds = ds
        rows = [r for r in common.gene_table()
                if r['dataset'] == ds and r['acc3'] is not None]
        self.rows = rows
        P = np.array([codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        self.model = sg.PairwiseRidge(alpha).fit(P, y)
        tab = {}
        for p, v in zip(P, y):
            tab.setdefault(tuple(int(x) for x in p), []).append(float(v))
        self.table = {k: float(np.mean(v)) for k, v in tab.items()}
        self.y_sd, self.y_mean = float(y.std()), float(y.mean())
        self.y_lo, self.y_hi = float(y.min()), float(y.max())
        self.n_fit = len(rows)

    def score(self, P, tabular=True):
        # The surrogate extrapolates above the measured maximum outside the sampled region,
        # so a benchmark score is an estimate on the accuracy scale, not an accuracy claim;
        # every result reported from this harness is a difference between configurations.
        P = np.asarray(P, int)
        out = self.model.predict(P)
        if tabular and self.table:
            for i, row in enumerate(P):
                v = self.table.get(tuple(int(x) for x in row))
                if v is not None:
                    out[i] = v
        return out

    def noise_for_rho(self, target, n=6000, seed=1):
        """Gaussian fitness noise whose achieved Spearman against truth equals `target`."""
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        P = rng.integers(0, CARD, (n, NG))
        t = self.score(P, tabular=False)
        lo, hi = 1e-6, 500.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if spearmanr(t + rng.normal(0, mid, n), t).statistic > target:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def predictor_emulator(self, n_bins=30, seed=0):
        """A fitness map that reproduces the pipeline's own LLM predictor empirically.

        The predictor's error is not Gaussian: it emits near-integer percentages and is
        strongly discretized. Binning the released architectures by their measured 3-epoch
        accuracy and resampling the predictor values observed in each bin reproduces both its
        rank fidelity and its tie structure.
        """
        rows = [r for r in self.rows if r.get('pred') is not None]
        a3 = np.array([r['acc3'] for r in rows])
        pr = np.array([r['pred'] for r in rows])
        e = np.quantile(a3, np.linspace(0, 1, n_bins + 1))
        e[0] -= 1e-9; e[-1] += 1e-9
        b = np.clip(np.digitize(a3, e) - 1, 0, n_bins - 1)
        pool = [pr[b == i] for i in range(n_bins)]
        rng = np.random.default_rng(seed)

        def f(t):
            sh = t.shape
            bb = np.clip(np.digitize(t.ravel(), e) - 1, 0, n_bins - 1)
            v = np.array([pool[i][rng.integers(0, len(pool[i]))] if len(pool[i]) else 0.0
                          for i in bb])
            return v.reshape(sh)
        return f


# -- running LLM-written operators --------------------------------------
class _TO(Exception):
    pass


#: exception classes a caller wants propagated out of an operator call rather than treated as
#: an operator failure (used for the per-operator CPU cap in s20_operator_bench.py)
FATAL_EXC = ()


def _h(s, f):
    raise _TO()


signal.signal(signal.SIGALRM, _h)


def load_operator(path):
    """Import one extracted operator file and return (kind, function).

    The extracted files carry a provenance comment at column 0 above a body that may itself
    be indented (the rewrites were spliced out of a class), so the comment lines are stripped
    before dedenting.
    """
    lines = open(path).read().splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith('#')
                     and not lines[0].startswith(' ')):
        lines.pop(0)
    c = textwrap.dedent('\n'.join(lines) + '\n')
    ns = {'random': random, 'math': math, 'np': np, 'numpy': np, 'copy': __import__('copy')}
    exec(compile(c, os.path.basename(path), 'exec'), ns)
    for k in ('mutate_gene', 'combine_genes', 'select_competitor', '_create_random_chromosome'):
        if callable(ns.get(k)):
            return k, ns[k]
    raise ValueError('no operator in ' + path)


class _MockGA:
    """What an operator sees: the search space, the mutation rate, the population."""

    def __init__(self):
        self.search_space = {k: list(v) for k, v in SS.items()}
        self.mutation_rate = 0.2
        self.population = []


def run_ga(bench, n_seg=1, gens=26, pop=20, elite=2, pm=0.2, k=3, R=200, seed=1,
           noise=0.0, ops=None, op_budget=0.05, record_genes=False, fitness_map=None,
           dup_penalty=False, timer_guard=True):
    """R replicates of an (n_seg x gens) GA on `bench`. Equal budget = n_seg*gens*pop.

    ops maps an operator name to a replacement function; a call that raises or exceeds
    `op_budget` seconds falls back to the handcrafted behavior, as the pipeline does.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    ops = ops or {}
    mock = _MockGA()
    fails = [0, 0]

    def _disarm():
        if timer_guard:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass

    def guarded(fn, *a, fallback=None):
        # The time-out signal can land between arming the timer and entering the try block, or
        # inside the finally that disarms it, so the whole body is wrapped a second time.
        fails[1] += 1
        try:
            if timer_guard:
                signal.setitimer(signal.ITIMER_REAL, op_budget)
            try:
                return fn(*a)
            except FATAL_EXC:
                raise
            except Exception:
                fails[0] += 1
                return fallback() if fallback else None
            finally:
                _disarm()
        except _TO:
            fails[0] += 1
            _disarm()
            return fallback() if fallback else None

    def _fit(t, P=None):
        f = fitness_map(t) if fitness_map is not None else t.copy()
        if noise:
            f = f + rng.normal(0, noise, f.shape)
        if dup_penalty and P is not None:
            for r in range(R):
                seen = set()
                for i in range(P.shape[1]):
                    key = P[r, i].tobytes()
                    if key in seen:
                        f[r, i] *= 0.5
                    seen.add(key)
        return f

    def init_pop(n):
        fn = ops.get('_create_random_chromosome')
        if fn is None:
            return rng.integers(0, CARD, (n, NG))
        out = np.empty((n, NG), int)
        for i in range(n):
            c = guarded(fn, mock, fallback=lambda: chrom_of(rng.integers(0, CARD)))
            if not isinstance(c, dict) or not set(GENES) <= set(c):
                c = chrom_of(rng.integers(0, CARD))
            out[i] = codes_of(c)
        return out

    def select(P, f, n_off):
        fn = ops.get('select_competitor')
        if fn is None:
            c = rng.integers(0, pop, (R, n_off, k))
            fv = np.take_along_axis(f[:, None, :].repeat(n_off, 1), c, 2)
            win = np.take_along_axis(c, fv.argmax(2)[:, :, None], 2)[:, :, 0]
            return np.take_along_axis(P, win[:, :, None].repeat(NG, 2), 1)
        out = np.empty((R, n_off, NG), int)
        for r in range(R):
            mock.population = [dict(chrom_of(P[r, i]), fitness=float(f[r, i]))
                               for i in range(pop)]
            for j in range(n_off):
                s = guarded(fn, mock, mock.population,
                            fallback=lambda: max(random.sample(mock.population, k),
                                                 key=lambda x: x['fitness']))
                if not isinstance(s, dict) or 'fitness' not in s:
                    s = max(random.sample(mock.population, k), key=lambda x: x['fitness'])
                out[r, j] = codes_of(s)
        return out

    def cross(A, B):
        fn = ops.get('combine_genes')
        pt = rng.integers(1, NG, (R, A.shape[1], 1))
        base = np.where(np.arange(NG)[None, None, :] < pt, A, B)
        if fn is None:
            return base
        out = base.copy()
        for r in range(R):
            for j in range(A.shape[1]):
                for gi, g in enumerate(GENES):
                    v = guarded(fn, mock, g, SS[g][A[r, j, gi]], SS[g][B[r, j, gi]],
                                int(pt[r, j, 0]), gi, NG, fallback=None)
                    if v is not None:
                        try:
                            out[r, j, gi] = code(g, v)
                        except Exception:
                            pass
        return out

    def mutate(P):
        fn = ops.get('mutate_gene')
        mm = rng.random(P.shape) < pm
        if fn is None:
            shift = rng.integers(1, CARD[None, None, :].repeat(P.shape[0], 0)
                                 .repeat(P.shape[1], 1))
            return np.where(mm, (P + shift) % CARD[None, None, :], P)
        out = P.copy()
        for r, j, gi in np.argwhere(mm):
            g = GENES[gi]; cur = SS[g][P[r, j, gi]]
            v = guarded(fn, mock, g, cur, list(SS[g]),
                        fallback=lambda: random.choice([x for x in SS[g] if x != cur]))
            if v is not None:
                try:
                    out[r, j, gi] = code(g, v)
                except Exception:
                    pass
        return out

    best = np.full(R, -np.inf)
    gen1, later, sampled, uniq = [], [], [], [set() for _ in range(R)]
    for _ in range(n_seg):
        P = np.stack([init_pop(pop) for _ in range(R)])
        t = bench.score(P.reshape(-1, NG)).reshape(R, pop)
        f = _fit(t)
        gen1.append(t.mean(1)); sampled.append(t)
        best = np.maximum(best, t.max(1))
        if record_genes:
            for r in range(R):
                uniq[r].update(map(tuple, P[r].tolist()))
        for _ in range(gens - 1):
            order = np.argsort(-f, 1)
            new = np.empty_like(P)
            new[:, :elite] = np.take_along_axis(P, order[:, :elite, None].repeat(NG, 2), 1)
            n_off = pop - elite
            new[:, elite:] = mutate(cross(select(P, f, n_off), select(P, f, n_off)))
            P = new
            t = bench.score(P.reshape(-1, NG)).reshape(R, pop)
            f = _fit(t, P)
            best = np.maximum(best, t.max(1))
            later.append(t.mean(1)); sampled.append(t)
            if record_genes:
                for r in range(R):
                    uniq[r].update(map(tuple, P[r].tolist()))
    S = np.concatenate(sampled, 1)
    res = dict(best=best, mean_sampled=S.mean(1), top10=np.sort(S, 1)[:, -10:].mean(1),
               gain_over_start=np.mean(later, 0) - np.mean(gen1, 0),
               op_fail_rate=fails[0] / max(1, fails[1]), op_calls=fails[1])
    if record_genes:
        res['n_unique'] = np.array([len(u) for u in uniq], float)
    return res
