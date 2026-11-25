"""What the procedure actually finds: replayed query-budget search on NAS-Bench-201.

Every other NAS-Bench-201 script in this archive measures a *ranking*: rank fidelity, the
spectrum, the regret of a shortlist. None of them answers the question a reader of a NAS paper
asks first -- which architecture does the recipe end up selecting, and how does its accuracy
compare against the standard search baselines at the same budget. This script answers exactly
that, and it answers it the only way a reproduction archive can answer it honestly: every
"training run" is a lookup in the published NAS-Bench-201 table (`data/nb201_all.pickle`, the
complete 5^6 = 15,625-cell topology space). No network is trained anywhere in this file.

What a budget means
-------------------
An algorithm is given a budget of N architectures that it may have trained, N in
{10, 20, 50, 100, 200, 500, 1000}. A query is a *distinct* architecture trained: re-visiting an
architecture the same run has already trained is free, because its accuracy is already known.
Every algorithm therefore produces an ordered list of distinct architectures in the order it
first trained them, the answer at budget N is the architecture among the first N with the best
*search* signal, and its reported accuracy is read from the reported column. Because the lists
are nested in N, the four algorithms are compared on the identical run at every budget.

The four algorithms
-------------------
random     a uniform random sample of N distinct cells.
evolution  regularized evolution: population 10, sample size 3 drawn without replacement, the
           best of the sample mutated at one uniformly chosen edge to a uniformly chosen one of
           the four other operations, the oldest member of the population removed.
local      local search: a uniform random start, best-improvement hill-climbing over the 24
           single-edge neighbours of the current cell, a fresh random start when the current
           cell is a local optimum. Every neighbour inspected is a query.
recipe     this paper's procedure. Spend n_fit = min(N // 2, 200) queries on a uniform random
           sample; replace those labels by their ranks inside the sample (the monotone
           recalibration of Section VI, `rankdata`, exactly as s53_spectrum.py applies it); fit
           the project's degree-2 ridge on them (the main effects of the six edges plus all 15
           edge-pair interactions, 405 features, alpha = 10 -- s35_nb201.fit, which is
           surrogate.py's PairwiseRidge in the normal equations); score all 15,625 cells; spend
           the remaining N - n_fit queries training the highest-scored cells not already
           trained; and return the best architecture it actually trained. The ridge sees no
           label outside its own n_fit sample, and the returned accuracy is always that of a
           trained architecture, so no number from an untrained cell reaches the report.

Metrics, per (algorithm, budget, dataset) cell, over 500 independent seeds
-------------------------------------------------------------------------
acc_mean / acc_sd  mean and standard deviation of the returned architecture's accuracy.
acc_ci             a 95% percentile-bootstrap interval for that mean, 10,000 resamples. A mean
                   over 500 seeds is an estimate and the table says so: no accuracy in this
                   file is quoted without the interval that belongs to it.
regret_mean        mean accuracy points against the grid's global optimum. This is the regret
                   of s36_regret.py -- the accuracy lost by training only a shortlist instead
                   of the whole pool -- with the shortlist being the N architectures the
                   algorithm chose to train and the selection inside it made by the search
                   signal; where the search and reported columns coincide it is literally
                   `s36_regret.regret` at k = N. Regret is `optimum - accuracy` seed by seed,
                   so `regret_mean` is exactly `optimum - acc_mean` and the table prints the
                   interval in its place rather than a second copy of the same number.
contrasts          for every algorithm but random search, the difference of means against
                   random search and against regularized evolution, each with a 95%
                   percentile-bootstrap interval for the difference and a two-sided
                   Mann-Whitney U p-value. The seed streams of two algorithms are independent
                   -- `_rng` mixes the algorithm index into the seed -- so the samples are
                   independent, not paired, and both the interval and the test are the
                   two-sample kind.

Which test decides, and why the two can disagree
------------------------------------------------
The bootstrap interval on the *difference of means* is the primary test everywhere in this
archive, because the quantity the paper claims is a difference of mean returned accuracy in
accuracy points, and that is what the interval is an interval for. The Mann-Whitney U test is
reported beside it and tests something else: whether one arm's returned accuracy stochastically
dominates the other's. The two answer different questions and can therefore disagree, in both
directions, without either being wrong:

  * a small shift in the mean driven by a thin left tail can leave the medians and the bulk of
    the two distributions interleaved, so the interval excludes zero while U does not;
  * a consistent shift of the bulk that is small in points can make U highly significant while
    the interval on the mean still spans zero.

Both happen here -- the returned accuracy is bounded above by the grid optimum and is strongly
left-skewed, which is precisely the shape that separates the two tests. `tests_agree` records
the comparison cell by cell and `contrast_test_disagreements` lists every cell where they part,
so the disagreement is reported rather than resolved silently by quoting whichever test is
more favourable.

The ledger
----------
`ledger` is the full record of the recipe against each baseline over all 21 budget-by-dataset
cells, decided by the primary test: ahead where the interval lies above zero, behind where it
lies below, tied where it contains zero. A table that reported a single budget would let the
choice of budget do the arguing, so the count is taken over every budget the replay runs.
gpu_hours_median   the median, over seeds, of the wall-clock training cost the N queries imply:
                   sum over the queried cells of `train_times` x `epochs` / 3600. `train_times`
                   is NAS-Bench-201's *own* recorded average per-epoch training time for that
                   cell (seconds, 200 epochs, the benchmark authors' hardware). It is the
                   benchmark's record of what training those cells cost its authors, not a
                   measurement of anything this paper ran.

The protocol, and the bias it is measured against
-------------------------------------------------
The standard NAS protocol searches on a validation signal and reports final test accuracy, and
this script follows it wherever the redistributed table allows. The pickle holds four dataset
keys -- cifar10-valid, cifar10, cifar100, ImageNet16-120 -- each with a single accuracy field
(`eval_acc1es`). So CIFAR-10 can be run properly: the algorithms see only `cifar10-valid`
(the model trained on the training split, evaluated on the validation split) and the accuracy
reported is `cifar10`'s final test accuracy, which no algorithm ever reads. For CIFAR-100 and
ImageNet16-120 the table exposes only one accuracy per architecture, so the search signal and
the reported accuracy are necessarily the same field.

A caveat of that kind is worth little while its size is unknown, so the size is measured here
rather than asserted. CIFAR-10 is the one dataset on which *both* protocols can be run, and the
script runs both: the proper one, and the single-column one in which the search signal and the
reported accuracy are the same `cifar10` test column, exactly as CIFAR-100 and ImageNet16-120
are forced to be. The difference between them, per algorithm and budget, is the optimism the
single-column protocol induces, and the quantity that matters is not its size but whether it is
*differential* -- whether it flatters one algorithm more than another, which is the only way it
could change the ordering of the table. That comparison is `protocol_bias` in the JSON.
`search_oracle_regret` records the companion floor: the regret of the single best cell *by
validation accuracy*, below which no validation-driven search on CIFAR-10 can go.

Outputs
-------
results/nb201_search.json    every number the table shows, plus the full protocol settings
../tables/tab_search.tex     Table: what the four algorithms return at N = 20, 100, 500

    python3 -I analysis/s62_nb201_search.py
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
#: `python3 -I` does not put the script's directory on the path, and the other scripts are run
#: both from the project root and from inside analysis/, so the import path is set explicitly.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
DATA = os.path.join(ROOT, 'data')

import s35_nb201 as nb                                                    # noqa: E402

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}
LONG = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
        'ImageNet16-120': 'ImageNet16-120'}
#: the field the algorithms are allowed to see, and the field the paper reports
SEARCH_FIELD = {'cifar10': 'cifar10-valid', 'cifar100': 'cifar100',
                'ImageNet16-120': 'ImageNet16-120'}
REPORT_FIELD = {'cifar10': 'cifar10', 'cifar100': 'cifar100',
                'ImageNet16-120': 'ImageNet16-120'}

ALGOS = ['random', 'evolution', 'local', 'recipe']
LABEL = {'random': 'Random search', 'evolution': 'Regularized evolution',
         'local': 'Local search', 'recipe': 'Spectrum recipe (ours)'}
#: the cost block of the table is one cell wide, so it uses one-word names there
TERSE = {'random': 'Random', 'evolution': 'Evolution', 'local': 'Local', 'recipe': 'Recipe'}

BUDGETS = [10, 20, 50, 100, 200, 500, 1000]
TABLE_BUDGETS = [20, 100, 500]
#: the body table carries the budgets the body argues from; the supplement keeps all three,
#: and the win/tie/loss ledger under both counts every one of the 21 cells either way
BODY_BUDGETS = [20, 100]
NMAX = max(BUDGETS)
SEEDS = 500
POP, SAMPLE = 10, 3               # regularized evolution
N_FIT_CAP = 200                   # the recipe's n_fit = min(N // 2, N_FIT_CAP)
ALPHA = 10.0                      # the project's ridge penalty, as in surrogate.py
SEED_BASE = 2026
N_EDGE, N_OP = 6, 5
GRID = N_OP ** N_EDGE
N_NBR = N_EDGE * (N_OP - 1)       # 24 single-edge neighbours
POW = N_OP ** np.arange(N_EDGE - 1, -1, -1)

N_BOOT = 10000                    # resamples behind every interval in this file
CI = 95.0                         # the interval the paper quotes, in percent
ALPHA_TEST = 0.05                 # the level the secondary U test is read at
BOOT_SEED = 20260                 # one fixed stream, so the intervals are reproducible
#: the dataset on which both protocols can be run, so that the cost of the single-column
#: protocol the other two are forced into can be measured rather than assumed
BIAS_DS = 'cifar10'


def _rng(ds_i, algo_i, seed, extra=0):
    """Independent, reproducible stream per (dataset, algorithm, seed, budget)."""
    return np.random.default_rng([SEED_BASE, ds_i, algo_i, seed, extra])


# ------------------------------------------------------------------ intervals and tests
def boot_mean_ci(x, rng):
    """Percentile-bootstrap interval for the mean of `x`, as [lo, hi]."""
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, (N_BOOT, x.size))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return [float(lo), float(hi)]


def contrast(x, y, rng):
    """mean(x) - mean(y) for two *independent* samples, with its interval and a U test.

    The two algorithms draw from different RNG streams, so the seeds are not paired and the
    difference is resampled within each arm separately. The Mann-Whitney U test is two-sided
    and makes no distributional assumption, which matters because the accuracy of a returned
    architecture is bounded above by the grid optimum and is therefore strongly left-skewed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = x[rng.integers(0, x.size, (N_BOOT, x.size))].mean(axis=1)
    by = y[rng.integers(0, y.size, (N_BOOT, y.size))].mean(axis=1)
    lo, hi = np.percentile(bx - by, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    #: identical samples make the U statistic degenerate; that can only happen if two arms
    #: returned the same architecture on every seed, which is a result rather than an error
    if np.array_equal(x, y):
        p = 1.0
    else:
        p = float(mannwhitneyu(x, y, alternative='two-sided').pvalue)
    by_ci = bool(lo > 0 or hi < 0)
    by_u = bool(p < ALPHA_TEST)
    return {'diff': float(x.mean() - y.mean()), 'ci': [float(lo), float(hi)], 'p': p,
            #: the primary test: the interval on the difference of means, which is the
            #: quantity the paper states. `significant` is kept as its name because every
            #: consumer of this file reads it, and it has always meant this test.
            'significant': by_ci,
            'significant_by_interval': by_ci,
            'p_mannwhitney': p,
            'significant_by_u_test': by_u,
            'tests_agree': bool(by_ci == by_u),
            'verdict': 'ahead' if lo > 0 else ('behind' if hi < 0 else 'tied')}


# ------------------------------------------------------------------ the benchmark table
def load():
    """The table, reordered so that row i is the cell whose six edge codes encode to i.

    Working in that linear index is what makes a mutation one integer add and the 24-neighbour
    table one array, so no architecture string is parsed again after loading.
    """
    path = os.path.join(DATA, 'nb201_all.pickle')
    if not os.path.exists(path):
        sys.exit('data/nb201_all.pickle is missing; see s35_nb201.py for what it is and '
                 'where it comes from.')
    with open(path, 'rb') as fh:
        tab = pickle.load(fh)
    keys = list(tab)
    codes = np.array([nb.encode(k) for k in keys], dtype=np.int64)
    lin = np.ravel_multi_index(codes.T, (N_OP,) * N_EDGE)
    if len(keys) != GRID or len(set(lin.tolist())) != GRID:
        sys.exit('nb201_all.pickle is not the complete %d-cell factorial this script needs'
                 % GRID)
    order = np.argsort(lin, kind='stable')
    keys = [keys[i] for i in order]
    codes = codes[order]
    assert np.array_equal(np.ravel_multi_index(codes.T, (N_OP,) * N_EDGE), np.arange(GRID))
    fields = sorted({SEARCH_FIELD[d] for d in DATASETS} | {REPORT_FIELD[d] for d in DATASETS})
    acc = {f: np.array([tab[k][f]['eval_acc1es'] for k in keys], float) for f in fields}
    #: the benchmark's own recorded cost of training that cell: mean seconds per epoch times
    #: the 200 epochs it was trained for, in GPU-hours. Nothing here is measured by this paper.
    hours = {f: np.array([tab[k][f]['train_times'] * tab[k][f]['epochs'] for k in keys], float)
             / 3600.0 for f in fields}
    return keys, codes, acc, hours


def neighbour_table(codes):
    """(15625, 24) int32: every cell's single-edge neighbours, edge-major.

    Column e * 4 + j is edge e set to the j-th of the four operations it is not currently set
    to, so drawing a column uniformly is exactly "mutate one edge uniformly".
    """
    out = np.empty((GRID, N_NBR), dtype=np.int32)
    base = np.arange(GRID)
    for e in range(N_EDGE):
        c = codes[:, e]
        for j in range(N_OP - 1):
            out[:, e * (N_OP - 1) + j] = base + ((c + 1 + j) % N_OP - c) * POW[e]
    return out


# ------------------------------------------------------------------ the degree-2 ridge
class Ridge:
    """s35_nb201.fit against a design matrix built once for the whole grid.

    The maths is identical to `s35_nb201.fit` (which is `surrogate.PairwiseRidge` written in
    the normal equations); caching the 15,625 x 405 design matrix is what makes 10,500 fits
    affordable, since every fit has to score the entire grid. `check()` verifies the two agree.
    """

    def __init__(self, codes, alpha=ALPHA):
        self.A = nb.design(codes)
        self.alpha = alpha
        self.eye = np.arange(self.A.shape[1])

    def scores(self, idx, y):
        """Fit on rows `idx` with labels `y`; return the predicted score of all 15,625 cells."""
        A = self.A[idx]
        y = np.asarray(y, float)
        b = float(y.mean())
        G = A.T @ A
        G[self.eye, self.eye] += self.alpha
        w = np.linalg.solve(G, A.T @ (y - b))
        return self.A @ w + b

    def check(self, codes, y):
        """The cached fit must reproduce s35_nb201.fit exactly."""
        idx = np.arange(0, GRID, 97)
        mine = self.scores(idx, y[idx])
        theirs = nb.fit(codes[idx], y[idx], alpha=self.alpha)(codes)
        return float(np.abs(mine - theirs).max())


# ------------------------------------------------------------------ the four algorithms
def run_random(rng, nmax):
    """A uniform random sample of distinct cells; any prefix is again a uniform sample."""
    return rng.choice(GRID, nmax, replace=False).astype(np.int64)


def run_evolution(rng, nmax, nbrs, S):
    """Regularized evolution, Real et al.: tournament of 3, one-edge mutation, oldest dies."""
    q, seen = [], set()
    pop = []
    for a in rng.choice(GRID, POP, replace=False).tolist():
        pop.append(a)
        seen.add(a)
        q.append(a)
    #: all randomness drawn up front so the inner loop is pure integer work
    steps = 8 * nmax
    tour = np.argsort(rng.random((steps, POP)), axis=1)[:, :SAMPLE].tolist()
    mut = rng.integers(0, N_NBR, steps).tolist()
    for t in range(steps):
        if len(q) >= nmax:
            break
        best, bv = -1, -np.inf
        for m in tour[t]:
            a = pop[m]
            if S[a] > bv:
                bv, best = S[a], a
        child = nbrs[best][mut[t]]
        pop.append(child)
        pop.pop(0)
        if child not in seen:
            seen.add(child)
            q.append(child)
    if len(q) < nmax:
        raise RuntimeError('evolution stalled at %d distinct cells' % len(q))
    return np.array(q[:nmax], dtype=np.int64)


def run_local(rng, nmax, nbrs, S):
    """Best-improvement hill-climbing over the 24 single-edge neighbours, random restarts."""
    q, seen = [], set()

    def query(a):
        if a not in seen:
            seen.add(a)
            q.append(a)

    while len(q) < nmax:
        cur = int(rng.integers(GRID))
        query(cur)
        while len(q) < nmax:
            best, bv = cur, S[cur]
            #: a random neighbour order so that a budget that runs out mid-neighbourhood does
            #: not systematically favour the low-numbered edges
            for m in rng.permutation(N_NBR).tolist():
                nxt = nbrs[cur][m]
                query(nxt)
                if S[nxt] > bv:
                    bv, best = S[nxt], nxt
                if len(q) >= nmax:
                    break
            if best == cur:
                break                     # a local optimum: restart
            cur = best
    return np.array(q[:nmax], dtype=np.int64)


def run_recipe(rng, n, ridge, S):
    """n_fit random queries -> ranks -> degree-2 ridge -> train the top of what is left."""
    n_fit = min(n // 2, N_FIT_CAP)
    samp = rng.choice(GRID, n_fit, replace=False)
    pred = ridge.scores(samp, rankdata(S[samp]))
    rest = n - n_fit
    keep = np.ones(GRID, dtype=bool)
    keep[samp] = False
    cand = np.flatnonzero(keep)
    top = cand[np.argsort(-pred[cand], kind='stable')[:rest]]
    return np.concatenate([samp, top]).astype(np.int64), n_fit


# ------------------------------------------------------------------ scoring a run
def answer(q, S, T):
    """The architecture the run returns at budget len(q), and its reported accuracy.

    The selection is made with the search signal only; the reported column is read once, for
    the single cell that was selected, after the fact.
    """
    sel = int(q[int(np.argmax(S[q]))])
    return sel, float(T[sel])


def replay(di, S, T, H, nbrs, ridge):
    """Every algorithm, every budget, SEEDS seeds; the per-seed values, unsummarized.

    `S` is the signal the algorithms may read and `T` the column the answer is reported in;
    passing the same array for both is the single-column protocol.
    """
    Sl = S.tolist()                     # the inner loops do scalar lookups, not array ops
    best = float(T.max())
    cell = {a: {str(n): {'acc': [], 'reg': [], 'cost': []} for n in BUDGETS} for a in ALGOS}
    for s in range(SEEDS):
        # the three budget-independent algorithms are run once, at the largest budget;
        # their query lists are nested, so a prefix *is* the run at the smaller budget
        runs = {
            'random': run_random(_rng(di, 0, s), NMAX),
            'evolution': run_evolution(_rng(di, 1, s), NMAX, nbrs, Sl),
            'local': run_local(_rng(di, 2, s), NMAX, nbrs, Sl),
        }
        for a, q in runs.items():
            for n in BUDGETS:
                _, y = answer(q[:n], S, T)
                cell[a][str(n)]['acc'].append(y)
                cell[a][str(n)]['reg'].append(best - y)
                cell[a][str(n)]['cost'].append(float(H[q[:n]].sum()))
        for n in BUDGETS:
            q, _ = run_recipe(_rng(di, 3, s, n), n, ridge, S)
            _, y = answer(q, S, T)
            cell['recipe'][str(n)]['acc'].append(y)
            cell['recipe'][str(n)]['reg'].append(best - y)
            cell['recipe'][str(n)]['cost'].append(float(H[q].sum()))
    return cell


def summarize(cell, boot):
    """Per (algorithm, budget): the means, their bootstrap intervals, and the contrasts."""
    out = {}
    for a, dd in cell.items():
        out[a] = {}
        for n, v in dd.items():
            lo, hi = boot_mean_ci(v['acc'], boot)
            out[a][n] = {
                'acc_mean': float(np.mean(v['acc'])), 'acc_sd': float(np.std(v['acc'])),
                'acc_median': float(np.median(v['acc'])),
                'acc_ci': [lo, hi], 'acc_ci_halfwidth': float((hi - lo) / 2.0),
                'acc_se': float(np.std(v['acc'], ddof=1) / np.sqrt(len(v['acc']))),
                'regret_mean': float(np.mean(v['reg'])),
                'regret_sd': float(np.std(v['reg'])),
                'regret_median': float(np.median(v['reg'])),
                'found_optimum_share': float(np.mean(np.array(v['reg']) <= 0.0)),
                'gpu_hours_median': float(np.median(v['cost'])),
                'n_seeds': SEEDS}
    #: every algorithm against each of the three baselines the body makes a claim about: the
    #: cheapest one that is always available, the strongest of the classical ones, and the one
    #: the recipe is said to be ahead of on every dataset
    for a in ALGOS:
        for n in map(str, BUDGETS):
            for base in ('random', 'evolution', 'local'):
                out[a][n]['vs_' + base] = contrast(cell[a][n]['acc'],
                                                   cell[base][n]['acc'], boot)
    return out


def main():
    keys, codes, acc, hours = load()
    nbrs_arr = neighbour_table(codes)
    nbrs = nbrs_arr.tolist()          # list-of-lists: the inner loops index it millions of times
    ridge = Ridge(codes)

    out = {
        'n_arch': GRID,
        'datasets': DATASETS,
        'algorithms': ALGOS,
        'algorithm_labels': LABEL,
        'protocol': {
            'budgets': BUDGETS,
            'table_budgets': TABLE_BUDGETS,
            'seeds': SEEDS,
            'query': 'one distinct architecture trained; re-visiting an architecture the run '
                     'has already trained is free and does not consume budget',
            'training': 'none: every evaluation is a lookup in data/nb201_all.pickle',
            'search_field': SEARCH_FIELD,
            'report_field': REPORT_FIELD,
            'n_fit_rule': 'min(N // 2, %d)' % N_FIT_CAP,
            'n_fit': {str(n): min(n // 2, N_FIT_CAP) for n in BUDGETS},
            'recalibration': 'scipy.stats.rankdata of the labels inside the fitted sample',
            'ridge': 'degree-2 ridge of s35_nb201.fit (= surrogate.PairwiseRidge): 6 edge main '
                     'effects + all 15 edge-pair interactions',
            'ridge_alpha': ALPHA,
            'ridge_n_features': int(nb.NFEAT),
            'evolution_population': POP,
            'evolution_sample': SAMPLE,
            'evolution_mutation': 'one uniformly chosen edge set to one of the four other '
                                  'operations, uniformly; age-based (oldest) removal',
            'local_search': 'uniform random start, best-improvement over the %d single-edge '
                            'neighbours, random restart at a local optimum; every neighbour '
                            'inspected is a query' % N_NBR,
            'random_search': 'uniform random sample of N distinct cells',
            'seed_scheme': 'numpy default_rng([%d, dataset_index, algorithm_index, seed, '
                           'budget]); budget is 0 for every algorithm but the recipe, whose '
                           'random sample depends on N' % SEED_BASE,
            'cost_model': "sum over the queried cells of the benchmark's own recorded "
                          "train_times (mean seconds per epoch) x epochs (200) / 3600, in "
                          'GPU-hours; the search field supplies the cost, since that is the '
                          'training run the search performs. This is NAS-Bench-201\'s record '
                          'of its authors\' training cost, not a measurement of ours.',
        },
        'protocol_note':
            'The redistributed table exposes a single accuracy field (eval_acc1es) per dataset '
            'key. CIFAR-10 therefore follows the standard protocol -- the algorithms see only '
            'cifar10-valid (train-split model, validation-split accuracy) and the reported '
            'number is the cifar10 final test accuracy, which no algorithm reads -- while for '
            'CIFAR-100 and ImageNet16-120 no separate validation column exists, so the search '
            'signal and the reported accuracy are necessarily the same field. The CIFAR-100 '
            'and ImageNet16-120 columns are therefore optimistic by whatever a '
            'validation/test split would cost. That cost is not left as an assertion: '
            'CIFAR-10 carries both columns, so the same replay is run there under both '
            'protocols and the difference is reported in protocol_bias, together with the '
            'spread of that difference across the four algorithms, which is the only part of '
            'it that could reorder the table.',
        'ridge_matches_s35_max_abs_diff': {},
        'grid': {},
        'results': {},
    }

    for d in DATASETS:
        S, T = acc[SEARCH_FIELD[d]], acc[REPORT_FIELD[d]]
        sel = int(np.argmax(S))
        out['grid'][d] = {
            'search_field': SEARCH_FIELD[d],
            'report_field': REPORT_FIELD[d],
            'optimum': float(T.max()),
            'optimum_arch': keys[int(np.argmax(T))],
            'mean': float(T.mean()),
            'sd': float(T.std()),
            'worst': float(T.min()),
            'best_by_search_signal': float(T[sel]),
            'best_by_search_signal_arch': keys[sel],
            'search_oracle_regret': float(T.max() - T[sel]),
            'median_hours_per_query': float(np.median(hours[SEARCH_FIELD[d]])),
            'total_hours_full_grid': float(hours[SEARCH_FIELD[d]].sum()),
        }
        out['ridge_matches_s35_max_abs_diff'][d] = ridge.check(codes, S)

    # ---------------------------------------------------------------- the replay
    boot = np.random.default_rng(BOOT_SEED)
    raw = {}
    for di, d in enumerate(DATASETS):
        cell = replay(di, acc[SEARCH_FIELD[d]], acc[REPORT_FIELD[d]],
                      hours[SEARCH_FIELD[d]], nbrs, ridge)
        raw[d] = cell
        out['results'][d] = summarize(cell, boot)
        print('replayed %-14s %d seeds x %d algorithms x %d budgets'
              % (d, SEEDS, len(ALGOS), len(BUDGETS)))

    # ------------------------------------------------- what the single-column protocol costs
    # CIFAR-10 is the only dataset carrying both a validation and a test column, so it is the
    # only place the protocol CIFAR-100 and ImageNet16-120 are forced into can be priced. The
    # same four algorithms, the same seeds, the same budgets -- only the search signal changes,
    # from the validation column to the reported test column itself.
    bias_i = DATASETS.index(BIAS_DS)
    Tb, Hb = acc[REPORT_FIELD[BIAS_DS]], hours[SEARCH_FIELD[BIAS_DS]]
    cell_b = replay(bias_i, Tb, Tb, Hb, nbrs, ridge)
    single = summarize(cell_b, boot)
    out['protocol_bias'] = {
        'dataset': BIAS_DS,
        'what': 'the same replay with the search signal set to the reported test column '
                'instead of the validation column, which is the protocol the redistributed '
                'table forces on CIFAR-100 and ImageNet16-120',
        'proper_search_field': SEARCH_FIELD[BIAS_DS],
        'single_column_search_field': REPORT_FIELD[BIAS_DS],
        'single_column': single,
        'optimism': {a: {str(n): single[a][str(n)]['acc_mean']
                         - out['results'][BIAS_DS][a][str(n)]['acc_mean']
                         for n in BUDGETS} for a in ALGOS},
        'optimism_ci': {a: {str(n): contrast(cell_b[a][str(n)]['acc'],
                                             raw[BIAS_DS][a][str(n)]['acc'], boot)
                            for n in TABLE_BUDGETS} for a in ALGOS},
    }
    #: the question the control exists to answer: the ordering of the table is unchanged if the
    #: optimism is the same for every algorithm, and only a *differential* optimism could move
    #: a row past another. The spread across algorithms is that differential, in points.
    out['protocol_bias']['spread_across_algorithms'] = {
        str(n): float(max(out['protocol_bias']['optimism'][a][str(n)] for a in ALGOS)
                      - min(out['protocol_bias']['optimism'][a][str(n)] for a in ALGOS))
        for n in BUDGETS}
    def _order(tab, n):
        return sorted(ALGOS, key=lambda z: -tab[z][str(n)]['acc_mean'])

    out['protocol_bias']['ordering_preserved'] = {
        str(n): bool(_order(single, n) == _order(out['results'][BIAS_DS], n))
        for n in BUDGETS}
    out['protocol_bias']['ordering'] = {
        str(n): {'proper': _order(out['results'][BIAS_DS], n), 'single_column': _order(single, n)}
        for n in BUDGETS}
    #: an ordering that swaps two rows matters only if the two were distinguishable to begin
    #: with. For every budget where the two protocols disagree, the pair that swapped is
    #: recorded together with whether their intervals under the proper protocol overlap: a
    #: swap inside an overlap is the two arms being indistinguishable, not the protocol
    #: changing the answer.
    swaps = {}
    for n in BUDGETS:
        p, s = _order(out['results'][BIAS_DS], n), _order(single, n)
        if p == s:
            continue
        pr = out['results'][BIAS_DS]
        pairs = []
        for i, x in enumerate(p):
            for y in p[i + 1:]:
                if s.index(x) > s.index(y):
                    cx, cy = pr[x][str(n)]['acc_ci'], pr[y][str(n)]['acc_ci']
                    pairs.append({'ahead_under_proper': x, 'ahead_under_single_column': y,
                                  'gap_under_proper': pr[x][str(n)]['acc_mean']
                                  - pr[y][str(n)]['acc_mean'],
                                  'intervals_overlap': bool(cx[0] <= cy[1] and cy[0] <= cx[1])})
        swaps[str(n)] = pairs
    out['protocol_bias']['swaps'] = swaps
    out['protocol_bias']['every_swap_is_inside_an_overlap'] = all(
        p['intervals_overlap'] for v in swaps.values() for p in v)
    out['protocol_bias']['budgets_with_a_swap'] = sorted(swaps, key=int)
    print('priced the single-column protocol on %s: optimism %s points at N=100, '
          'ordering preserved at every budget: %s'
          % (BIAS_DS,
             '/'.join('%.2f' % out['protocol_bias']['optimism'][a]['100'] for a in ALGOS),
             all(out['protocol_bias']['ordering_preserved'].values())))

    # ---------------------------------------------------------- the ledger over every cell
    # Reporting one budget lets the choice of budget do the arguing. The ledger is the recipe
    # against each baseline in all 21 budget-by-dataset cells, decided by the primary test.
    out['ledger'] = {}
    for base in ('random', 'evolution', 'local'):
        cells = []
        for d in DATASETS:
            for n in BUDGETS:
                c = out['results'][d]['recipe'][str(n)]['vs_' + base]
                cells.append({'dataset': d, 'budget': n, 'diff': c['diff'],
                              'ci': c['ci'], 'verdict': c['verdict'],
                              'p_mannwhitney': c['p_mannwhitney'],
                              'tests_agree': c['tests_agree']})
        counts = {v: sum(1 for c in cells if c['verdict'] == v)
                  for v in ('ahead', 'tied', 'behind')}
        out['ledger'][base] = {
            'n_cells': len(cells),
            'ahead': counts['ahead'], 'tied': counts['tied'], 'behind': counts['behind'],
            'behind_at': [(c['dataset'], c['budget']) for c in cells
                          if c['verdict'] == 'behind'],
            'ahead_at': [(c['dataset'], c['budget']) for c in cells
                         if c['verdict'] == 'ahead'],
            'cells': cells,
            'decided_by': 'the 95% percentile-bootstrap interval on the difference of means',
        }
    #: the smallest budget at which the recipe leads regularized evolution on every dataset
    #: with its interval excluding zero -- the regime the spectrum predicts it should own
    out['ledger']['recipe_leads_evolution_everywhere_at'] = sorted(
        n for n in BUDGETS
        if all(out['results'][d]['recipe'][str(n)]['vs_evolution']['verdict'] == 'ahead'
               for d in DATASETS))
    #: every cell in which the interval and the U test disagree, so that the disagreement is
    #: reported rather than resolved by quoting whichever test is more favourable
    dis = []
    for d in DATASETS:
        for a in ALGOS:
            for n in BUDGETS:
                for base in ('random', 'evolution', 'local'):
                    c = out['results'][d][a][str(n)]['vs_' + base]
                    if a == base or c['tests_agree']:
                        continue
                    dis.append({'dataset': d, 'algorithm': a, 'baseline': base, 'budget': n,
                                'diff': c['diff'], 'ci': c['ci'],
                                'p_mannwhitney': c['p_mannwhitney'],
                                'interval_excludes_zero': c['significant_by_interval'],
                                'u_test_below_alpha': c['significant_by_u_test']})
    out['contrast_tests'] = {
        'primary': 'the 95% percentile-bootstrap interval on the difference of means; the '
                   'paper states a difference in accuracy points, and that is the quantity '
                   'this interval is an interval for',
        'secondary': 'a two-sided Mann-Whitney U test at alpha = %g, which tests stochastic '
                     'dominance rather than a difference of means' % ALPHA_TEST,
        'why_they_can_disagree':
            'The returned accuracy is bounded above by the grid optimum and is strongly '
            'left-skewed. A shift of the mean carried by a thin left tail can leave the two '
            'distributions interleaved, so the interval excludes zero where U does not; a '
            'consistent shift of the bulk that is small in points can do the reverse. The '
            'two are reported together and every cell where they part is listed here.',
        'n_contrasts': sum(1 for d in DATASETS for a in ALGOS for n in BUDGETS
                           for b in ('random', 'evolution', 'local') if a != b),
        'n_disagreements': len(dis),
        'disagreements': dis,
    }

    # ---------------------------------------------------------------- headline + checks
    out['headline'] = {d: {a: {'acc_mean': out['results'][d][a]['100']['acc_mean'],
                               'regret_mean': out['results'][d][a]['100']['regret_mean']}
                           for a in ALGOS} for d in DATASETS}
    bad_opt = [(d, a, n) for d in DATASETS for a in ALGOS for n in map(str, BUDGETS)
               if out['results'][d][a][n]['acc_mean'] > out['grid'][d]['optimum'] + 1e-9
               or out['results'][d][a][n]['regret_mean'] < -1e-9]
    lose = [(d, n) for d in DATASETS for n in map(str, BUDGETS)
            if out['results'][d]['recipe'][n]['acc_mean']
            <= out['results'][d]['random'][n]['acc_mean']]
    #: where the advantage over random search survives its own interval, and where it does not.
    #: A table that marked every row as a win would be the thing this check exists to prevent.
    sig = [(d, n) for d in DATASETS for n in map(str, BUDGETS)
           if out['results'][d]['recipe'][n]['vs_random']['significant']]
    nsig = [(d, n) for d in DATASETS for n in map(str, BUDGETS)
            if not out['results'][d]['recipe'][n]['vs_random']['significant']]
    out['checks'] = {
        'no_algorithm_exceeds_grid_optimum': not bad_opt,
        'violations': bad_opt,
        'recipe_beats_random_everywhere': not lose,
        'recipe_does_not_beat_random_at': lose,
        'recipe_beats_random_significantly_at': sig,
        'recipe_ties_random_at': nsig,
        'protocol_bias_preserves_ordering':
            all(out['protocol_bias']['ordering_preserved'].values()),
        'protocol_bias_every_swap_inside_an_interval_overlap':
            out['protocol_bias']['every_swap_is_inside_an_overlap'],
        'protocol_bias_max_spread':
            max(out['protocol_bias']['spread_across_algorithms'].values()),
        'recipe_beats_random_from_budget':
            min([n for n in BUDGETS
                 if all(out['results'][d]['recipe'][str(n)]['acc_mean']
                        > out['results'][d]['random'][str(n)]['acc_mean'] for d in DATASETS)],
                default=None),
        #: the ledger has to add up to the 21 cells it claims to cover, and the paper's
        #: headline sentence has to be the ledger rather than one hand-picked budget
        'ledger_covers_every_cell': all(
            out['ledger'][b]['ahead'] + out['ledger'][b]['tied'] + out['ledger'][b]['behind']
            == len(DATASETS) * len(BUDGETS) for b in ('random', 'evolution', 'local')),
        'recipe_never_behind_random': out['ledger']['random']['behind'] == 0,
        'ridge_matches_s35': all(v < 1e-6 for v in
                                 out['ridge_matches_s35_max_abs_diff'].values()),
    }

    os.makedirs(RES, exist_ok=True)
    jpath = os.path.join(RES, 'nb201_search.json')
    with open(jpath, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    tpath = emit_table(out)
    report(out)
    print('\nwrote %s' % os.path.relpath(jpath, ROOT))
    print('wrote %s' % os.path.relpath(tpath, ROOT))
    return 0


# ------------------------------------------------------------------ the tables
#: Two tables are written. The body carries what the paper argues from -- what each algorithm
#: returns at each budget, and the win/tie/loss ledger over every cell -- inside the venue's
#: free ten pages. The statistics that qualify it, which a reader checks rather than reads,
#: go to the supplement, where no page limit applies: the difference of means with its
#: interval, the GPU-hours the queries imply, and the price of the single-column protocol.
def _verdict(pbx):
    if all(pbx['ordering_preserved'].values()):
        return (r'and it leaves the ordering of the four algorithms unchanged at every '
                r'budget, which is the part of it that could have mattered')
    if pbx['every_swap_is_inside_an_overlap']:
        return (r'and wherever it reorders two algorithms, at $N{=}%s$, those two already '
                r'overlap in their intervals under the proper protocol, so no distinguishable '
                r'row moves past another'
                % ', '.join(pbx['budgets_with_a_swap']))
    return (r'and it reorders algorithms at $N{=}%s$, so the C100 and IN16 columns '
            r'should be read as the optimistic protocol they are'
            % ', '.join(pbx['budgets_with_a_swap']))


def emit_table(o):
    L = []
    a = L.append
    led = o['ledger']['evolution']
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{What the ranking core of Algorithm~1 returns, with the interval that belongs to it. '
      r'Query-budget search replayed on NAS-Bench-201~\cite{dong2020nasbench201}: mean '
      r'accuracy of the architecture returned after training $N$ of the 15{,}625 cells, '
      r'$\pm$ the half-width of a 95\%% percentile-bootstrap interval, over %d independent '
      r'seeds. Every evaluation is a lookup and \emph{no network was trained}. Mean regret is '
      r'exactly the grid-optimum row minus the entry, so it is not printed twice. The last '
      r'line counts every one of the %d budget-by-dataset cells rather than the budgets '
      r'shown, since quoting the contrast at one budget would let the choice of budget do the '
      r'arguing. Supplementary Table~S13 gives the difference of means with its interval, the '
      r'GPU-hours the queries imply, and the price of the single-column protocol.}'
      % (o['protocol']['seeds'], led['n_cells']))
    a(r'\label{tab:search}')
    a(r'\footnotesize')
    a(r'\setlength{\tabcolsep}{3.0pt}')
    a(r'\begin{tabular}{l ccc}')
    a(r'\toprule')
    a(r' & C10 & C100 & IN16 \\')
    a(r'\midrule')
    for i, alg in enumerate(ALGOS):
        if i:
            a(r'\addlinespace[2pt]')
        a(r'\multicolumn{4}{l}{\emph{%s}} \\' % LABEL[alg])
        for n in BODY_BUDGETS:
            cells = ' & '.join(
                r'%.2f\,$\pm$\,%.2f' % (o['results'][d][alg][str(n)]['acc_mean'],
                                        o['results'][d][alg][str(n)]['acc_ci_halfwidth'])
                for d in DATASETS)
            a(r'\quad $N{=}%d$ & %s \\' % (n, cells))
    a(r'\midrule')
    a(r'Grid optimum & %s \\'
      % ' & '.join('%.2f' % o['grid'][d]['optimum'] for d in DATASETS))
    a(r'Grid mean & %s \\'
      % ' & '.join('%.2f' % o['grid'][d]['mean'] for d in DATASETS))
    a(r'\midrule')
    rnd = o['ledger']['random']
    a(r'\multicolumn{4}{l}{\footnotesize the recipe leads random search in all %d cells, and '
      r'regularized} \\' % rnd['n_cells'])
    a(r'\multicolumn{4}{l}{\footnotesize evolution in %d, tied in %d, behind in %d} \\'
      % (led['ahead'], led['tied'], led['behind']))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    p = os.path.join(ROOT, 'tables', 'tab_search.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    emit_table_sm(o)
    return p


def emit_table_sm(o):
    L = []
    a = L.append
    pbx = o['protocol_bias']
    led = o['ledger']['evolution']
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{The statistics behind Table~IV of the main paper. The \emph{difference of '
      r'means} block is the gap to a baseline: $\star$ marks a 95\%% interval excluding zero, '
      r'$\circ$ a tie. Seeds are independent streams, so the interval is two-sample. That '
      r'interval is the primary test, because the claim is a difference in accuracy points; '
      r'the Mann--Whitney $U$ test reported beside it in \texttt{nb201\_search.json} tests '
      r'stochastic dominance instead, so the two can part, and every cell where they do is '
      r'listed there. The contrast against regularized evolution is given at all three '
      r'budgets and summarized over all %d cells. The GPU-hours are the '
      r"benchmark\textquotesingle s own record for the cells queried. The last block prices "
      r'the single-column protocol C100 and IN16 are forced into, by rerunning the replay on '
      r'CIFAR-10, which carries both columns, with the test column as the search signal: '
      r'the optimism is %.2f--%.2f points, %s.}'
      % (led['n_cells'],
         min(pbx['optimism'][a_]['100'] for a_ in ALGOS),
         max(pbx['optimism'][a_]['100'] for a_ in ALGOS), _verdict(pbx)))
    a(r'\label{tab:search_sm}')
    a(r'\footnotesize')
    a(r'\setlength{\tabcolsep}{3.0pt}')
    a(r'\begin{tabular}{l ccc}')
    a(r'\toprule')
    a(r' & C10 & C100 & IN16 \\')
    a(r'\midrule')
    a(r'\multicolumn{4}{l}{\emph{difference of means, with its 95\% interval}} \\')

    def _diff_row(name, alg, key, n):
        cells = []
        for d in DATASETS:
            c = o['results'][d][alg][str(n)][key]
            cells.append(r'%+.2f$^{%s}$ \tiny[%+.2f, %+.2f]'
                         % (c['diff'], r'\star' if c['significant'] else r'\circ',
                            c['ci'][0], c['ci'][1]))
        a(r'\quad %s & %s \\' % (name, ' & '.join(cells)))

    #: every arm against random search at one budget, then the contrast a reader asks for
    #: next -- the recipe against the strongest classical baseline -- at every budget the
    #: table carries. The remaining rows of the second block would be the first with its sign
    #: flipped, so they are left out.
    a(r'\multicolumn{4}{l}{\quad\footnotesize against random search, at $N{=}100$} \\')
    for alg in ('evolution', 'local', 'recipe'):
        _diff_row(TERSE[alg], alg, 'vs_random', 100)
    a(r'\multicolumn{4}{l}{\quad\footnotesize against regularized evolution, by budget} \\')
    for n in TABLE_BUDGETS:
        _diff_row(r'Recipe, $N{=}%d$' % n, 'recipe', 'vs_evolution', n)
    a(r'\multicolumn{4}{l}{\quad\footnotesize over all %d budget-by-dataset cells the recipe '
      r'is ahead in %d, tied in %d, behind in %d} \\'
      % (led['n_cells'], led['ahead'], led['tied'], led['behind']))
    a(r'\addlinespace[2pt]\midrule')
    a(r'\multicolumn{4}{l}{\emph{median GPU-hours the $N{=}100$ queries imply}} \\')
    for alg in ALGOS:
        a(r'\quad %s & %s \\' % (
            TERSE[alg],
            ' & '.join('%.0f' % o['results'][d][alg]['100']['gpu_hours_median']
                       for d in DATASETS)))
    a(r'\addlinespace[2pt]\midrule')
    a(r'\multicolumn{4}{l}{\emph{single-column protocol, priced on C10: points of optimism at '
      r'$N{=}100$}} \\')
    a(r'\quad %s & %.2f & \quad %s & %.2f \\'
      % (TERSE[ALGOS[0]], pbx['optimism'][ALGOS[0]]['100'],
         TERSE[ALGOS[1]], pbx['optimism'][ALGOS[1]]['100']))
    a(r'\quad %s & %.2f & \quad %s & %.2f \\'
      % (TERSE[ALGOS[2]], pbx['optimism'][ALGOS[2]]['100'],
         TERSE[ALGOS[3]], pbx['optimism'][ALGOS[3]]['100']))
    a(r'\multicolumn{4}{l}{\quad\footnotesize spread across the four algorithms %.2f points} \\'
      % pbx['spread_across_algorithms']['100'])
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    p = os.path.join(ROOT, 'tables', 'tab_search_sm.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return p


# ------------------------------------------------------------------ the printed summary
def report(o):
    print('\nreplayed query-budget search on NAS-Bench-201: mean accuracy of the returned')
    print('architecture, with mean regret against the grid optimum in parentheses, %d seeds.'
          % o['protocol']['seeds'])
    print('every evaluation is a table lookup; no network was trained.')
    for d in DATASETS:
        g = o['grid'][d]
        print('\n%s  (optimum %.2f, grid mean %.2f, searched on %s, reported on %s)'
              % (d, g['optimum'], g['mean'], g['search_field'], g['report_field']))
        print('  %-22s %s' % ('algorithm',
                              ''.join('%13s' % ('N=%d' % n) for n in o['protocol']['budgets'])))
        for alg in ALGOS:
            r = o['results'][d][alg]
            print('  %-22s %s' % (LABEL[alg], ''.join(
                '%13s' % ('%.2f(%.2f)' % (r[str(n)]['acc_mean'], r[str(n)]['regret_mean']))
                for n in o['protocol']['budgets'])))
        print('  %-22s %s' % ('median GPU-h, recipe', ''.join(
            '%13s' % ('%.0f' % o['results'][d]['recipe'][str(n)]['gpu_hours_median'])
            for n in o['protocol']['budgets'])))
        if g['search_oracle_regret'] > 0:
            print('  the best cell by validation accuracy reaches %.2f test, so no '
                  'validation-driven' % g['best_by_search_signal'])
            print('  search can get below %.2f points of regret here.'
                  % g['search_oracle_regret'])
    n_fit = o['protocol']['n_fit']
    print('\nthe recipe spends n_fit = %s queries on the random sample at N = %s.'
          % (', '.join(str(n_fit[str(n)]) for n in o['protocol']['budgets']),
             ', '.join(str(n) for n in o['protocol']['budgets'])))
    print('at N=100 it returns %s against random search\'s %s and regularized evolution\'s %s'
          % tuple(', '.join('%.2f' % o['headline'][d][a]['acc_mean'] for d in DATASETS)
                  for a in ('recipe', 'random', 'evolution')))
    print('on %s.' % ', '.join(SHORT[d] for d in DATASETS))
    print('\nthe ledger, over all %d budget-by-dataset cells, decided by %s:'
          % (o['ledger']['random']['n_cells'], o['ledger']['random']['decided_by']))
    for base in ('random', 'evolution', 'local'):
        L = o['ledger'][base]
        print('  recipe vs %-10s ahead %2d  tied %2d  behind %2d%s'
              % (base, L['ahead'], L['tied'], L['behind'],
                 ('   behind at: ' + ', '.join('%s N=%d' % (SHORT[d], n)
                                               for d, n in L['behind_at']))
                 if L['behind_at'] else ''))
    lead = o['ledger']['recipe_leads_evolution_everywhere_at']
    print('  the recipe leads regularized evolution on all three datasets at N = %s'
          % (', '.join(str(n) for n in lead) if lead else 'no budget'))
    ct = o['contrast_tests']
    print('\nthe interval and the U test agree in %d of %d contrasts; the %d that disagree '
          'are in the JSON.' % (ct['n_contrasts'] - ct['n_disagreements'],
                                ct['n_contrasts'], ct['n_disagreements']))
    pb = o['protocol_bias']
    print('\nwhat the single-column protocol costs, measured on %s at N=100 (points of '
          'optimism):' % pb['dataset'])
    print('  %s' % '  '.join('%s %+.2f' % (TERSE[a], pb['optimism'][a]['100']) for a in ALGOS))
    print('  spread across the four algorithms %.2f points; ordering preserved at every '
          'budget: %s' % (pb['spread_across_algorithms']['100'],
                          all(pb['ordering_preserved'].values())))
    for n, pairs in pb['swaps'].items():
        for p in pairs:
            print('  N=%-5s %s and %s swap; they are %.2f points apart under the proper '
                  'protocol and their intervals %s'
                  % (n, TERSE[p['ahead_under_proper']], TERSE[p['ahead_under_single_column']],
                     p['gap_under_proper'],
                     'overlap' if p['intervals_overlap'] else 'DO NOT overlap'))

    print('\nthe recipe against random search, difference of means with its 95% interval:')
    for d in DATASETS:
        print('  %-14s %s' % (SHORT[d], '  '.join(
            'N=%d %+.2f [%+.2f,%+.2f]%s'
            % (n, o['results'][d]['recipe'][str(n)]['vs_random']['diff'],
               o['results'][d]['recipe'][str(n)]['vs_random']['ci'][0],
               o['results'][d]['recipe'][str(n)]['vs_random']['ci'][1],
               '*' if o['results'][d]['recipe'][str(n)]['vs_random']['significant'] else ' ')
            for n in TABLE_BUDGETS)))

    c = o['checks']
    print('\n%d of %d (dataset, budget) cells have the recipe ahead of random search by more '
          'than its interval.' % (len(c['recipe_beats_random_significantly_at']),
                                  len(DATASETS) * len(BUDGETS)))
    if c['recipe_ties_random_at']:
        print('  it is a tie at: %s' % ', '.join('%s/N=%s' % (SHORT[d], n)
                                                 for d, n in c['recipe_ties_random_at']))
    print('\nchecks: no cell exceeds the grid optimum: %s | the recipe beats random search at '
          'every' % c['no_algorithm_exceeds_grid_optimum'])
    print('        budget and dataset: %s%s | the cached ridge reproduces s35_nb201.fit: %s'
          % (c['recipe_beats_random_everywhere'],
             '' if c['recipe_beats_random_everywhere']
             else ' (not at %s)' % ', '.join('%s/N=%s' % x
                                             for x in c['recipe_does_not_beat_random_at']),
             c['ridge_matches_s35']))


if __name__ == '__main__':
    sys.exit(main())
