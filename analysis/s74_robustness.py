"""Two robustness checks on the exact NAS-Bench-201 spectrum.

1. Functionally equivalent cells. Many of the 15,625 labelled cells compute the same function:
   an edge carrying `none` contributes nothing, an edge that lies on no path from the input
   node to the output node contributes nothing, and when the two intermediate nodes are not
   joined they are interchangeable. Each cell is reduced to a canonical form under those three
   rules, every cell of a class is given its class's mean accuracy, and the exact spectrum is
   recomputed. The grid stays a complete factorial, so the decomposition stays exact; what
   changes is that architectures the benchmark trained separately but which compute the same
   function no longer carry different values.

2. The rank transform as a null. Ranking gives every task the same uniform marginal, so it is
   fair to ask whether ranking forces spectra together regardless of the function. Synthetic
   functions are built on the same 5^6 grid with prescribed, deliberately different spectra
   (one concentrated at low order, one carrying most of its variance at orders three and four),
   compressed against a ceiling by a monotone map, and ranked. If ranking merely equalized
   spectra, the two families would land on one value; if it undoes a compression, each lands
   back near its own.

3. The scope test at small label counts. Step 4 of Algorithm 1 falls back to a proxy when the
   ridge's held-out R^2 is not positive. On so few labels that R^2 is itself noisy, so it is
   measured here where the ridge is known to work (NAS-Bench-201): five-fold cross-validated
   R^2 of the recalibrated ridge within the labelled sample, at 5 to 200 labels, 200 draws each.

Writes results/robustness.json.
"""
import itertools, json, os

import numpy as np
from scipy.stats import rankdata

import common
import s53_spectrum as s53

# edge order of the architecture string: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3)
EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
NONE = 0                       # s53.OPS[0] == 'none'


def canonical(g):
    """A key shared by every cell that computes the same function."""
    live = {(u, v): op for (u, v), op in zip(EDGES, g) if op != NONE}
    # keep only edges on some path from node 0 to node 3
    fwd, bwd = {0}, {3}
    for u, v in sorted(live):
        if u in fwd:
            fwd.add(v)
    for u, v in sorted(live, reverse=True):
        if v in bwd:
            bwd.add(u)
    on_path = {e: op for e, op in live.items() if e[0] in fwd and e[1] in bwd}
    if not on_path:
        return ('no-path',)
    key = tuple(sorted(on_path.items()))
    if (1, 2) not in on_path:          # nodes 1 and 2 are interchangeable
        swap = {1: 2, 2: 1, 0: 0, 3: 3}
        alt = tuple(sorted(((swap[u], swap[v]) if swap[u] < swap[v] else (swap[v], swap[u]), op)
                           for (u, v), op in on_path.items()))
        key = min(key, alt)
    return key


def spectrum(G, y):
    comp, Y = s53.anova(G, y)
    _, var = s53.by_order(comp, Y.shape)
    tot = sum(var.values())
    return {k: v / tot for k, v in var.items()}


def cumulative(sp, d):
    return float(sum(sp[k] for k in sp if k <= d))


def equivalence(G, acc):
    keys = [canonical(g) for g in G]
    classes = {}
    for i, k in enumerate(keys):
        classes.setdefault(k, []).append(i)
    out = {'n_cells': len(G), 'n_classes': len(classes),
           'largest_class': max(len(v) for v in classes.values()), 'datasets': {}}
    for d in s53.DATASETS:
        y = acc[d]
        ym = np.empty_like(y)
        for idx in classes.values():
            ym[idx] = y[idx].mean()
        within = float(np.mean([y[idx].var() for idx in classes.values() if len(idx) > 1]))
        row = {}
        for name, v in (('raw', y), ('class_mean', ym)):
            for scale, z in (('acc', v), ('recal', rankdata(v).astype(float))):
                sp = spectrum(G, z)
                row['%s_%s' % (name, scale)] = {'le1': cumulative(sp, 1), 'le2': cumulative(sp, 2),
                                                'le3': cumulative(sp, 3)}
        row['within_class_variance_share'] = within / float(y.var())
        out['datasets'][d] = row
    return out


def synthetic(G, shares, rng):
    """A function on the grid with the prescribed variance share at each order."""
    shape = (s53.N_OP,) * s53.N_EDGE
    f = np.zeros(len(G))
    for k, w in shares.items():
        if w <= 0:
            continue
        part = np.zeros(shape)
        for S in itertools.combinations(range(s53.N_EDGE), k):
            sh = [s53.N_OP if a in S else 1 for a in range(s53.N_EDGE)]
            part = part + rng.normal(size=sh)
        comp, _ = s53.anova(G, part[tuple(G.T)])
        pure = np.zeros(shape)
        for S in itertools.combinations(range(s53.N_EDGE), k):
            pure = pure + np.broadcast_to(comp[S], shape)
        v = pure[tuple(G.T)]
        f = f + np.sqrt(w) * v / v.std()
    return f


def compress(f, strength):
    """A ceiling: the best architectures squeezed together, as on an easy task."""
    z = (f - f.mean()) / f.std()
    return -np.exp(-strength * z)


def null_check(G, seeds=10):
    families = {'low_order': {1: 0.55, 2: 0.35, 3: 0.10},
                'high_order': {1: 0.15, 2: 0.15, 3: 0.35, 4: 0.35}}
    out = {}
    for name, shares in families.items():
        rows = []
        for s in range(seeds):
            f = synthetic(G, shares, np.random.default_rng(100 + s))
            r = {'raw': cumulative(spectrum(G, f), 2)}
            for st in (1.0, 2.0):
                c = compress(f, st)
                r['compressed_%g' % st] = cumulative(spectrum(G, c), 2)
                r['compressed_%g_ranked' % st] = cumulative(spectrum(G, rankdata(c).astype(float)), 2)
            r['ranked'] = cumulative(spectrum(G, rankdata(f).astype(float)), 2)
            rows.append(r)
        out[name] = {'prescribed_le2': float(sum(v for k, v in shares.items() if k <= 2)),
                     **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                     'seeds': seeds}
    return out


def scope_test(draws=200, ns=(5, 10, 25, 50, 100, 200)):
    import s35_nb201 as s35
    _, P, y, _ = s35.load()
    out = {}
    for d in s35.DS:
        out[d] = {}
        for n in ns:
            r2s = []
            for seed in range(draws):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(P), n, replace=False)
                tr = rankdata(y[d][idx]).astype(float)
                k = min(5, n)
                fold = np.arange(n) % k
                rng.shuffle(fold)
                pred = np.empty(n)
                for f in range(k):
                    te = fold == f
                    pred[te] = s35.fit(P[idx][~te], tr[~te])(P[idx][te])
                r2s.append(1 - ((tr - pred) ** 2).sum() / ((tr - tr.mean()) ** 2).sum())
            r2s = np.array(r2s)
            out[d][str(n)] = {'median_r2': float(np.median(r2s)), 'share_not_positive': float((r2s <= 0).mean())}
    return out


def main():
    G, acc = s53.load()
    res = {'equivalence': equivalence(G, acc), 'rank_null': null_check(G), 'scope_test': scope_test()}
    json.dump(res, open(os.path.join(common.RES, 'robustness.json'), 'w'), indent=1)
    e = res['equivalence']
    print('equivalence classes: %d of %d cells (largest %d)' % (e['n_classes'], e['n_cells'], e['largest_class']))
    for d, row in e['datasets'].items():
        print('  %-15s order<=2 raw %.1f%% -> class means %.1f%%;  recalibrated %.1f%% -> %.1f%%;  '
              'within-class variance %.1f%% of total'
              % (d, 100 * row['raw_acc']['le2'], 100 * row['class_mean_acc']['le2'],
                 100 * row['raw_recal']['le2'], 100 * row['class_mean_recal']['le2'],
                 100 * row['within_class_variance_share']))
    for d, v in res['scope_test'].items():
        print('  scope test %-15s' % d + '  '.join('n=%s: R2<=0 in %.0f%%' % (n, 100 * x['share_not_positive']) for n, x in v.items()))
    for name, r in res['rank_null'].items():
        print('  %-10s prescribed %.2f  raw %.3f  ranked %.3f  compressed(2) %.3f -> ranked %.3f'
              % (name, r['prescribed_le2'], r['raw'], r['ranked'], r['compressed_2'], r['compressed_2_ranked']))


if __name__ == '__main__':
    main()
