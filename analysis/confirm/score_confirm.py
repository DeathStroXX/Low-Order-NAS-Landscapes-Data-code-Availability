"""Score the pre-registered confirmation run against the predictions it was registered with.

This script embodies the decision rule of `PREREGISTRATION.md` so that the rule is applied by
code rather than by judgement after the fact. It reads the logs the confirmation run writes
and prints, for each claim and dataset, the measured difference, its bootstrap interval and
the verdict the pre-registration assigns -- confirmed, directionally confirmed, refuted or
unresolved.

It is runnable before the run exists: with no logs it prints the registered predictions and
the thresholds, which is the whole content of the registration.

Usage
-----
    python3 score_confirm.py [logdir]

`logdir` holds one JSON file per arm per dataset per seed, named
`<claim>_<arm>_<dataset>_<seed>.json`, each with at least::

    {"claim": "C1", "arm": "recipe", "dataset": "cifar10", "seed": 4001,
     "outcome_acc3": 83.9, "epochs_used": 100, "n_trained": 73,
     "n_discarded_at_floor": 9}

Outputs `results/confirm.json` beside the other machine-readable results.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')

#: the registered predictions, from Tables V and VI. Do not edit after the run.
CLAIMS = {
    'C1': {'section': 'VIII', 'table': 'S4',
           'treatment': 'recipe', 'control': 'flat',
           'budget': '100 training epochs',
           'predicted': {'cifar10': 4.28, 'cifar100': 7.95}},
    'C2': {'section': 'VIII', 'table': 'S5',
           'treatment': 'sitl-seeded', 'control': 'train-all',
           'budget': '120 trained architectures',
           'predicted': {'cifar10': 4.18, 'cifar100': 4.76}},
}
#: the mechanism check of Sec. VIII: the cold loop is predicted to lose outright.
MECHANISM = {'claim': 'C2', 'treatment': 'sitl-cold', 'control': 'train-all',
             'predicted': {'cifar10': -7.01, 'cifar100': -8.52}}
SEEDS = [4001, 4002, 4003]
HALF = 0.5          #: "at least half the predicted value"
NBOOT = 10000


def load(logdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(logdir, '*.json'))):
        with open(f) as fh:
            r = json.load(fh)
        need = {'claim', 'arm', 'dataset', 'seed', 'outcome_acc3'}
        if not need <= set(r):
            print('skipping %s: missing %s' % (os.path.basename(f), sorted(need - set(r))))
            continue
        rows.append(r)
    return rows


def dry(rows):
    """True when any log came from the dry-run evaluator.

    `run_confirm.py --evaluator bench` exercises the policy code on the surrogate benchmark and writes
    logs in this format. That landscape is the one the policies were priced on and it is
    smooth by construction, so a result computed from it cannot be evidence for them. The
    verdicts are still printed, because the dry run is how the protocol code is checked, but
    they are labelled and the registration's language is withheld.
    """
    return any(r.get('dry_run') or r.get('evaluator') not in (None, 'gpu') for r in rows)


def boot(x, seed=0):
    x = np.asarray(x, float)
    g = np.random.default_rng(seed)
    m = [np.mean(x[g.integers(0, len(x), len(x))]) for _ in range(NBOOT)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(diff, ci, predicted):
    """The decision rule of the pre-registration, applied verbatim."""
    lo, hi = ci
    if lo <= 0 <= hi:
        return 'unresolved'
    if hi < 0:
        return 'refuted' if predicted > 0 else 'confirmed'
    if predicted < 0:
        return 'refuted'
    return 'confirmed' if diff >= HALF * predicted else 'directional only'


def contrast(rows, claim, treatment, control, ds, predicted):
    def pick(arm):
        return {r['seed']: r['outcome_acc3'] for r in rows
                if r['claim'] == claim and r['arm'] == arm and r['dataset'] == ds}
    a, b = pick(treatment), pick(control)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {'status': 'no data', 'predicted': predicted,
                'threshold': HALF * predicted, 'seeds': []}
    d = [a[s] - b[s] for s in seeds]
    ci = boot(d)
    return {'status': 'measured', 'seeds': seeds, 'n': len(seeds),
            'treatment_mean': float(np.mean([a[s] for s in seeds])),
            'control_mean': float(np.mean([b[s] for s in seeds])),
            'diff': float(np.mean(d)), 'diff_per_seed': d, 'ci': list(ci),
            'predicted': predicted, 'threshold': HALF * predicted,
            'verdict': verdict(float(np.mean(d)), ci, predicted),
            'missing_seeds': sorted(set(SEEDS) - set(seeds))}


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'logs')
    rows = load(logdir) if os.path.isdir(logdir) else []
    is_dry = dry(rows)
    # record the log folder relative to the archive, never as a machine-specific absolute path
    root = os.path.dirname(os.path.dirname(HERE))
    absdir = os.path.abspath(logdir)
    shown = (os.path.relpath(absdir, root) if absdir.startswith(root + os.sep)
             else os.path.basename(absdir.rstrip(os.sep)))
    out = {'logdir': shown, 'n_log_files': len(rows), 'seeds_registered': SEEDS,
           'rule': 'see analysis/confirm/PREREGISTRATION.md section 2',
           'dry_run': is_dry,
           'half_threshold': HALF, 'n_boot': NBOOT, 'claims': {}}
    for name, c in CLAIMS.items():
        out['claims'][name] = {'section': c['section'], 'table': c['table'],
                               'budget': c['budget'],
                               'treatment': c['treatment'], 'control': c['control'],
                               'by_dataset': {
                                   ds: contrast(rows, name, c['treatment'], c['control'],
                                                ds, p)
                                   for ds, p in c['predicted'].items()}}
    out['mechanism'] = {
        'what': 'Sec. VIII predicts the loop without a prior loses outright',
        'by_dataset': {ds: contrast(rows, MECHANISM['claim'], MECHANISM['treatment'],
                                    MECHANISM['control'], ds, p)
                       for ds, p in MECHANISM['predicted'].items()}}

    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'confirm.json'), 'w'), indent=1)

    if not rows:
        print('No confirmation logs found in %s.\n'
              'The registration stands as follows; run the protocol of PREREGISTRATION.md '
              'and rerun this script.\n' % logdir)
    elif is_dry:
        print('DRY RUN. These logs come from run_confirm.py --evaluator bench, which '
              'exercises the\nprotocol on the surrogate benchmark -- the same surrogate landscape the '
              'policies were priced on.\nThe verdicts below check that the code applies the '
              'registered rule; NONE of them is a\nconfirmation of anything, and the paper '
              'quotes no number from a dry run.\n')
    for name, c in out['claims'].items():
        print('%s (Sec. %s, Table %s): %s vs %s at %s'
              % (name, c['section'], c['table'], c['treatment'], c['control'], c['budget']))
        for ds, e in c['by_dataset'].items():
            if e['status'] == 'no data':
                print('   %-9s predicted %+.2f, confirmed only if >= %+.2f with an interval '
                      'excluding zero' % (ds, e['predicted'], e['threshold']))
            else:
                v = e['verdict'].upper()
                if is_dry:
                    v = 'DRY RUN (%s)' % v.lower()
                print('   %-9s measured %+.2f [%+.2f, %+.2f] against %+.2f predicted  ->  %s'
                      % (ds, e['diff'], e['ci'][0], e['ci'][1], e['predicted'], v))
                if e['missing_seeds']:
                    print('             missing seeds %s' % e['missing_seeds'])
    print('\nwrote %s' % os.path.join(RES, 'confirm.json'))


if __name__ == '__main__':
    main()
