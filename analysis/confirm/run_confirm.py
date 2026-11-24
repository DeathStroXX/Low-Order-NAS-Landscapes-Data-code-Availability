"""Execute the pre-registered confirmation run of PREREGISTRATION.md.

`score_confirm.py` applies the registered decision rule; this script produces the logs that
rule reads. The two are kept apart on purpose: the policies live here, the verdict lives
there, and neither can be adjusted to suit the other after the fact.

What it runs
------------
The seven arms of the registration, unchanged and with no argument that could alter them:

  C1, at a budget of 100 training epochs per arm per dataset per seed
    flat    draw 100 candidates uniformly, train each one epoch, return the best
    screen  screen 800 free candidates with the surrogate, train the best 100 for one epoch
    halve   73 candidates for one epoch, the top quarter for a second, half of those for a third
    recipe  screen 800 free, then allocate the 100 epochs by halving as in `halve`

  C2, at a budget of 120 trained architectures per arm per dataset per seed
    train-all    the GA of the released pipeline, every individual trained, 6 generations of 20
    sitl-seeded  population 200 ranked by the free model, the best 4 per generation trained, 30
                 generations, the model seeded with the surrogate transferred from the *other*
                 dataset and refitted on the residual of what has been measured
    sitl-cold    the same loop with no prior, fitted only on what it measures

Seeds 4001, 4002, 4003 for every arm. The surrogate's ridge penalty is the alpha = 10 of
`surrogate.py`, untouched. No label of a run's own dataset enters its screen.

Where the accuracies come from
------------------------------
The policies are separated from the thing that evaluates an architecture, because the
registration is a statement about the policies and the evaluator is what costs GPU-hours:

  --evaluator gpu    the real run. Trains each architecture with the released pipeline's
                     recipe and returns its measured 3-epoch accuracy. This is the arm of the
                     registration that costs 24 GPU-hours, and it needs the training harness,
                     which is not importable without a GPU and the datasets. The entry point it
                     expects is `train_architecture(chrom, dataset, seed) -> acc3 or None`,
                     with None meaning the architecture fell below the pipeline's accuracy
                     floor and was discarded; the discard rate is logged per arm, because the
                     released distributions are left-truncated by it.
  --evaluator bench  a dry run on the surrogate benchmark. It exercises every line of the policy code and
                     writes logs in the same format, which is how the protocol was checked
                     before any GPU time was spent. It is NOT the confirmation: the surrogate
                     landscape is smooth by construction and the policies were priced on it, so
                     a dry run cannot be evidence for them. `score_confirm.py` marks any result
                     produced this way as a dry run and refuses to call it a confirmation.

Usage
-----
    python3 run_confirm.py --evaluator bench --out logs_dry     # protocol check, no GPU
    python3 run_confirm.py --evaluator gpu  --out logs          # the registered run

One JSON per arm per dataset per seed is written to `--out`, named
`<claim>_<arm>_<dataset>_<seed>.json`, which is what `score_confirm.py` reads.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, '..')
sys.path.insert(0, ANALYSIS)

import surrogate as sg          # noqa: E402
from surrogate import CARD, NG  # noqa: E402

#: fixed by the registration; none of these is a command-line argument
SEEDS = [4001, 4002, 4003]
DATASETS = ['cifar10', 'cifar100']
OTHER = {'cifar10': 'cifar100', 'cifar100': 'cifar10'}
C1_EPOCHS = 100
C1_SCREEN = 800
C1_FLAT_N = 100
C1_HALVE = (73, 0.25, 0.5)      # candidates, promote fraction to 2 epochs, then to 3
C2_TRAINED = 120
C2_POP, C2_GENS, C2_TRAIN_PER_GEN = 200, 30, 4
C2_GA_POP, C2_GA_GENS = 20, 6
ALPHA = 10.0


# --------------------------------------------------------------------------- evaluators
class GPUEvaluator:
    """The real thing: train an architecture and return its measured 3-epoch accuracy."""

    kind = 'gpu'

    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed
        self.trained = 0
        self.discarded = 0
        try:
            from train_harness import train_architecture       # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                'the GPU evaluator needs the training harness of the released pipeline, '
                'which exposes train_architecture(chrom, dataset, seed) and is not importable '
                'here (%s).\n'
                'Run the protocol check first with --evaluator bench, then run this on the '
                'machine that has the harness and the datasets.' % exc)
        self._train = train_architecture

    def __call__(self, P, epochs):
        out = np.full(len(P), np.nan)
        for i, chrom in enumerate(P):
            acc = self._train(sg.chrom_of(chrom), self.dataset, self.seed, epochs=epochs)
            self.trained += 1
            if acc is None:
                self.discarded += 1
            else:
                out[i] = acc
        return out


class BenchEvaluator:
    """A dry run on the surrogate benchmark. Exercises the policies; proves nothing about them."""

    kind = 'bench'

    def __init__(self, dataset, seed):
        import fractalbench
        self.bench = fractalbench.Bench(dataset, alpha=ALPHA)
        self.rng = np.random.default_rng(seed)
        self.dataset, self.seed = dataset, seed
        self.trained = 0
        self.discarded = 0
        # A shorter training run is a noisier estimate of the 3-epoch reference; the two
        # fidelities below are the ones the released record measures for 1 and 2 epochs.
        self._sd = {1: self.bench.noise_for_rho(0.923, seed=seed),
                    2: self.bench.noise_for_rho(0.951, seed=seed), 3: 0.0}

    def __call__(self, P, epochs):
        self.trained += len(P)
        y = self.bench.score(P)
        sd = self._sd.get(epochs, 0.0)
        return y + (self.rng.normal(0, sd, len(y)) if sd else 0.0)


def truth(evaluator, P):
    """The registered outcome: the 3-epoch reference accuracy of the returned architecture."""
    v = evaluator(np.atleast_2d(P), 3)
    return float(np.nanmax(v))


# --------------------------------------------------------------------------- the surrogate
def transferred_model(dataset):
    """A degree-2 ridge fitted on the released record of the OTHER dataset only."""
    import common
    rows = [r for r in common.gene_table()
            if r['dataset'] == OTHER[dataset] and r['acc3'] is not None]
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows])
    return sg.PairwiseRidge(ALPHA).fit(P, y)


def draw(rng, n):
    return rng.integers(0, CARD, (n, NG))


# --------------------------------------------------------------------------- C1 policies
def c1_flat(ev, rng, model):
    P = draw(rng, C1_FLAT_N)
    y = ev(P, 1)
    return P[int(np.nanargmax(y))], C1_FLAT_N


def c1_screen(ev, rng, model):
    S = draw(rng, C1_SCREEN)
    keep = S[np.argsort(-model.predict(S))[:C1_FLAT_N]]
    y = ev(keep, 1)
    return keep[int(np.nanargmax(y))], C1_FLAT_N


def _halve(ev, P):
    n, f2, f3 = C1_HALVE
    y1 = ev(P, 1)
    k2 = max(1, int(round(len(P) * f2)))
    P2 = P[np.argsort(-np.nan_to_num(y1, nan=-1e9))[:k2]]
    y2 = ev(P2, 2)
    k3 = max(1, int(round(k2 * f3)))
    P3 = P2[np.argsort(-np.nan_to_num(y2, nan=-1e9))[:k3]]
    y3 = ev(P3, 3)
    used = len(P) * 1 + k2 * 1 + k3 * 1
    return P3[int(np.nanargmax(y3))], used


def c1_halve(ev, rng, model):
    return _halve(ev, draw(rng, C1_HALVE[0]))


def c1_recipe(ev, rng, model):
    S = draw(rng, C1_SCREEN)
    keep = S[np.argsort(-model.predict(S))[:C1_HALVE[0]]]
    return _halve(ev, keep)


C1_ARMS = {'flat': c1_flat, 'screen': c1_screen, 'halve': c1_halve, 'recipe': c1_recipe}


# --------------------------------------------------------------------------- C2 policies
def c2_train_all(ev, rng, model):
    """The released pipeline's GA with every individual trained."""
    P = draw(rng, C2_GA_POP)
    y = ev(P, 3)
    best = (P[int(np.nanargmax(y))], float(np.nanmax(y)))
    n = C2_GA_POP
    for _ in range(C2_GA_GENS - 1):
        if n >= C2_TRAINED:
            break
        order = np.argsort(-np.nan_to_num(y, nan=-1e9))
        elite = P[order[:2]]
        kids = []
        while len(kids) < C2_GA_POP - 2:
            a, b = (P[order[rng.integers(0, 3)]] for _ in range(2))
            cut = rng.integers(1, NG)
            c = np.concatenate([a[:cut], b[cut:]])
            m = rng.random(NG) < 0.2
            c[m] = rng.integers(0, CARD)[m]
            kids.append(c)
        P = np.vstack([elite, np.array(kids)])[: min(C2_GA_POP, C2_TRAINED - n)]
        y = ev(P, 3)
        n += len(P)
        if np.nanmax(y) > best[1]:
            best = (P[int(np.nanargmax(y))], float(np.nanmax(y)))
    return best[0], n


def _sitl(ev, rng, prior):
    """Wide population ranked by the model, the best few trained, the model refitted.

    With a prior, the model is the fixed transferred surrogate plus a ridge refitted each
    generation on the residual of everything measured so far, which is what the registration
    specifies. With no prior it is a ridge on the measured set alone, which is the cold arm.
    """
    seenP, seeny = [], []
    corr = None
    best = (None, -np.inf)
    n = 0
    for _ in range(C2_GENS):
        if n >= C2_TRAINED:
            break
        P = draw(rng, C2_POP)
        if prior is not None:
            rank = prior.predict(P) + (corr.predict(P) if corr is not None else 0.0)
        elif seenP:
            rank = sg.PairwiseRidge(ALPHA).fit(np.vstack(seenP),
                                               np.concatenate(seeny)).predict(P)
        else:
            rank = rng.random(len(P))
        take = min(C2_TRAIN_PER_GEN, C2_TRAINED - n)
        keep = P[np.argsort(-rank)[:take]]
        y = ev(keep, 3)
        n += take
        if not np.isfinite(y).any():
            continue
        y = np.nan_to_num(y, nan=float(np.nanmin(y)))
        seenP.append(keep)
        seeny.append(y)
        if float(np.max(y)) > best[1]:
            best = (keep[int(np.argmax(y))], float(np.max(y)))
        if prior is not None:
            X, t = np.vstack(seenP), np.concatenate(seeny)
            corr = sg.PairwiseRidge(ALPHA).fit(X, t - prior.predict(X))
    return best[0], n


def c2_sitl_seeded(ev, rng, model):
    return _sitl(ev, rng, model)


def c2_sitl_cold(ev, rng, model):
    return _sitl(ev, rng, None)


C2_ARMS = {'train-all': c2_train_all, 'sitl-seeded': c2_sitl_seeded,
           'sitl-cold': c2_sitl_cold}


# --------------------------------------------------------------------------- driver
def run_one(claim, arm, fn, dataset, seed, evaluator_cls, outdir):
    ev = evaluator_cls(dataset, seed)
    rng = np.random.default_rng(seed)
    model = transferred_model(dataset)
    chrom, used = fn(ev, rng, model)
    rec = {
        'claim': claim, 'arm': arm, 'dataset': dataset, 'seed': int(seed),
        'outcome_acc3': truth(ev, chrom),
        'epochs_used': int(used) if claim == 'C1' else None,
        'n_trained': int(ev.trained),
        'n_discarded_at_floor': int(ev.discarded),
        'evaluator': ev.kind,
        'dry_run': ev.kind != 'gpu',
        'architecture': [int(x) for x in np.asarray(chrom).ravel()],
    }
    path = os.path.join(outdir, '%s_%s_%s_%d.json' % (claim, arm, dataset, seed))
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    print('  %-12s %-9s seed %d -> %.2f  (%d trained, %d discarded)'
          % (arm, dataset, seed, rec['outcome_acc3'], ev.trained, ev.discarded))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--evaluator', choices=['gpu', 'bench'], required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'logs'))
    a = ap.parse_args()
    cls = {'gpu': GPUEvaluator, 'bench': BenchEvaluator}[a.evaluator]
    os.makedirs(a.out, exist_ok=True)

    if a.evaluator == 'bench':
        print('DRY RUN on the surrogate benchmark. This checks the protocol code; it is not the')
        print('confirmation, and score_confirm.py will not read it as one.\n')

    for claim, arms in (('C1', C1_ARMS), ('C2', C2_ARMS)):
        print('%s:' % claim)
        for arm, fn in arms.items():
            for ds in DATASETS:
                for seed in SEEDS:
                    run_one(claim, arm, fn, ds, seed, cls, a.out)
        print()
    print('wrote %d logs to %s' % (len(os.listdir(a.out)), a.out))
    print('now: python3 analysis/confirm/score_confirm.py %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
