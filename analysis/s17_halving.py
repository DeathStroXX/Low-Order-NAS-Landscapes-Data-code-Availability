"""Step 3 -- how to spend a training budget: successive halving against a flat epoch count.

Section VII of the original manuscript recommended two epochs per candidate. That is
dominated: train every candidate for one epoch, promote a fraction to a second epoch and
half of those to a third, and the three-epoch reference's own choice is recovered for well
under two thirds of its cost. The learning curves released with the paper make this
measurable without retraining anything.

Writes results/halving.json, tables/tab_halving.tex.
"""
import json, os, csv
import numpy as np
import common

RNG = np.random.default_rng(11)
T = 4000
POOL = 100


def curves():
    """The recovered curves, keyed by architecture rather than by evaluation.

    The released file has one row per (architecture, run, epoch). A handful of architectures
    were evaluated in more than one run, so the number of architecture-run *pairs* in the file
    and the number of *distinct* architectures it covers are two different counts, and the
    policies below draw from the distinct set. Both are returned, because the supplement
    quotes both and they must not be allowed to drift into each other.
    """
    path = os.path.join(common.DATA, 'learning_curves.csv')
    by, pairs, uids, nrows = {}, set(), set(), 0
    for rec in csv.DictReader(open(path)):
        k = (rec['uid'], rec['dataset'])
        by.setdefault(k, {})[int(rec['epoch'])] = float(rec['test_acc'])
        pairs.add((rec['dataset'], rec['method'], rec['run'], rec['uid']))
        uids.add(k)
        nrows += 1
    out = {}
    for (uid, ds), e in by.items():
        if {1, 2, 3} <= set(e):
            out.setdefault(ds, []).append((e[1], e[2], e[3]))
    counts = {'architecture_run_pairs': len(pairs), 'distinct_architectures': len(uids),
              'rows': nrows,
              'note': 'a pair is one architecture in one run; the policies draw from the '
                      'distinct set, so the smaller count is the one they are measured on'}
    return {ds: np.array(v) for ds, v in out.items()}, counts


def main():
    C, counts = curves()
    out = {'curve_counts': counts}
    for ds, M in C.items():
        e1, e2, e3 = M[:, 0], M[:, 1], M[:, 2]
        n = len(M)
        d = RNG.integers(0, n, (T, POOL))
        cells = {}

        def rec(name, acc, cost):
            cells[name] = dict(acc=float(np.mean(acc)), sd=float(np.std(acc) / np.sqrt(T)),
                               epochs=int(round(cost)), frac=cost / (3 * POOL))
        ref = e3[d].max(1)
        rec('full3', ref, 3 * POOL)
        rec('epoch1', e3[d[np.arange(T), e1[d].argmax(1)]], POOL)
        rec('epoch2', e3[d[np.arange(T), e2[d].argmax(1)]], 2 * POOL)
        for frac in (0.5, 0.25, 0.1):
            k = max(2, int(POOL * frac)); k2 = max(1, k // 2)
            keep = np.take_along_axis(d, np.argsort(-e1[d], 1)[:, :k], 1)
            keep2 = np.take_along_axis(keep, np.argsort(-e2[keep], 1)[:, :k2], 1)
            acc = e3[keep2[np.arange(T), e3[keep2].argmax(1)]]
            rec(f'halving_{int(frac*100)}', acc, POOL + k + k2)
        # regret and hit@3 against the pool's own three-epoch optimum
        for k, c in cells.items():
            c['regret'] = cells['full3']['acc'] - c['acc']
        # hit@3: does the policy return one of the pool's three best?
        def hit(sel):
            top3 = np.sort(e3[d], 1)[:, -3:]
            return float(np.mean([s >= t.min() for s, t in zip(sel, top3)]))
        cells['epoch1']['hit3'] = hit(e3[d[np.arange(T), e1[d].argmax(1)]])
        cells['epoch2']['hit3'] = hit(e3[d[np.arange(T), e2[d].argmax(1)]])
        k = 25; k2 = 12
        keep = np.take_along_axis(d, np.argsort(-e1[d], 1)[:, :k], 1)
        keep2 = np.take_along_axis(keep, np.argsort(-e2[keep], 1)[:, :k2], 1)
        cells['halving_25']['hit3'] = hit(e3[keep2[np.arange(T), e3[keep2].argmax(1)]])
        cells['full3']['hit3'] = 1.0
        out[ds] = dict(n_curves=n, cells=cells)
    counts['distinct_in_policies'] = sum(v['n_curves'] for k, v in out.items()
                                         if k != 'curve_counts')
    if counts['distinct_in_policies'] != counts['distinct_architectures']:
        raise SystemExit('the policies drew from %d curves but the file covers %d distinct '
                         'architectures' % (counts['distinct_in_policies'],
                                            counts['distinct_architectures']))
    json.dump(out, open(os.path.join(common.RES, 'halving.json'), 'w'), indent=1)
    _tex({k: v for k, v in out.items() if k != 'curve_counts'})
    print('%d rows: %d architecture-run pairs, %d distinct architectures'
          % (counts['rows'], counts['architecture_run_pairs'],
             counts['distinct_architectures']))
    for ds, o in out.items():
        if ds == 'curve_counts':
            continue
        print(f"\n== {ds}  ({o['n_curves']} complete curves)")
        for k, c in o['cells'].items():
            h = f"  hit@3 {c['hit3']*100:3.0f}%" if 'hit3' in c else ''
            print(f"   {k:12s} {c['epochs']:4d} ep ({c['frac']*100:3.0f}% of full)  "
                  f"acc {c['acc']:6.2f}  regret {c['regret']:+.2f}{h}")


def _tex(out):
    rows = [('full3', 'Three epochs on every candidate'),
            ('epoch2', 'Two epochs on every candidate'),
            ('epoch1', 'One epoch on every candidate'),
            ('halving_50', 'Halving, promote top 50\\%'),
            ('halving_25', 'Halving, promote top 25\\%'),
            ('halving_10', 'Halving, promote top 10\\%')]
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         '& \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         '\\textbf{Budget policy} & \\textbf{Cost} & \\textbf{Regret} & \\textbf{Cost} '
         '& \\textbf{Regret} \\\\', '\\midrule']
    for k, lab in rows:
        a, b = out['cifar10']['cells'][k], out['cifar100']['cells'][k]
        L.append(f"{lab} & {a['frac']*100:.0f}\\% & {a['regret']:.2f} & "
                 f"{b['frac']*100:.0f}\\% & {b['regret']:.2f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_halving.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
