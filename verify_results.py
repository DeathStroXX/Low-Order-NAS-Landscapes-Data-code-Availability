"""Re-run the analyses in a temporary copy of this archive and compare with the committed results.

    python3 verify_results.py            the quick steps (about 5 minutes)
    python3 verify_results.py --all      also the long ones (model-in-the-loop search, the warm
                                         start and the trained predictors: about 25 minutes more)

Nothing in your checkout is modified: the archive is copied to a temporary directory, the
scripts run there, and every result file they write is compared value by value with the
committed one under analysis/results/. Steps whose third-party input is missing (see
data/external/PROVENANCE.md) are reported as skipped, not as failures. The replayed
NAS-Bench-201 search (s62, hours) is never run here. Exit status 0 when everything that ran
matches.
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
QUICK = [('s02_check_record', []), ('s14_learning_curves', []), ('s10_fitness_provenance', []),
         ('s11_proxy_fidelity', []), ('s72_censoring', []), ('s5_predictor_validation', []),
         ('s16_surrogate', []), ('s15_prescreen', []), ('s27_signal_anatomy', []),
         ('s25_transfer', []), ('s36_regret', []), ('s17_halving', []), ('s30_recipe', []),
         ('s73_supplement_extras', []),
         ('s53_spectrum', ['data/nb201_all.pickle']), ('s59_ceiling', ['data/nb201_all.pickle']),
         ('s57_benchmarks', ['data/external/nas-bench-macro_cifar10.json']),
         ('s58_hardware', ['data/external/HW-NAS-Bench-v1_0.pickle']),
         ('s56_targets', ['data/nb201_all.pickle']), ('s35_nb201', ['data/nb201_all.pickle']),
         ('s74_robustness', ['data/nb201_all.pickle']),
         ('s75_k_sensitivity', ['data/nb201_all.pickle']),
         ('s63_extra_grids', ['data/external/nats-bench-sss.csv'])]
LONG = [('s31_sitl', []), ('s29_warmstart', []), ('s68_learned_predictors', ['data/nb201_all.pickle'])]


def diff(a, b, path='', out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append('%s/%s: present in only one' % (path, k))
            else:
                diff(a[k], b[k], '%s/%s' % (path, k), out, tol)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, '%s[%d]' % (path, i), out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol * max(1.0, abs(a)):
            out.append('%s: %r -> %r' % (path, a, b))
    elif a != b:
        out.append('%s: %r -> %r' % (path, str(a)[:60], str(b)[:60]))
    return out


def main():
    steps = QUICK + (LONG if '--all' in sys.argv else [])
    tmp = tempfile.mkdtemp(prefix='verify_')
    work = os.path.join(tmp, 'archive')
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    committed = os.path.join(ROOT, 'analysis', 'results')
    ran, skipped, bad = 0, [], []
    for name, needs in steps:
        if any(not os.path.exists(os.path.join(ROOT, n)) for n in needs):
            skipped.append(name); continue
        before = {f: os.path.getmtime(os.path.join(work, 'analysis', 'results', f))
                  for f in os.listdir(os.path.join(work, 'analysis', 'results'))}
        r = subprocess.run([sys.executable, name + '.py'], cwd=os.path.join(work, 'analysis'),
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad.append('%s exited with status %d: %s' % (name, r.returncode, r.stderr.strip()[-300:]))
            continue
        ran += 1
        for f in os.listdir(os.path.join(work, 'analysis', 'results')):
            if f.endswith('.json') and os.path.getmtime(os.path.join(work, 'analysis', 'results', f)) != before.get(f):
                d = diff(json.load(open(os.path.join(committed, f))),
                         json.load(open(os.path.join(work, 'analysis', 'results', f))), f)
                bad += ['%s (%s): %s' % (name, f, x) for x in d[:5]]
        print('  %-26s ok' % name, flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print('\n%d steps re-run, %d skipped (third-party input not fetched: %s)'
          % (ran, len(skipped), ', '.join(skipped) or 'none'))
    if bad:
        print('%d DIFFERENCES:' % len(bad)); [print('  ' + b) for b in bad[:40]]
        return 1
    print('every result that was recomputed matches the committed one')
    return 0


if __name__ == '__main__':
    sys.exit(main())
