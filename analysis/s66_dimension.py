"""The dimension of the degree-$d$ model class, and the condition that makes it identifiable.

Proposition 1 bounds what a degree-$d$ function can fit. The companion question -- how many
labels it takes to *find* that function -- is answered by the dimension of the class, not by
the size of the space, and the two differ by many orders of magnitude on the grids this paper
measures. The main paper states that as a corollary; this script computes its numbers.

For a product grid with factor cardinalities q_1..q_m, the ANOVA component of a subset S of
factors has one free value per combination of the *contrasts* of the factors in S, since every
component is constrained to average to zero along each axis it depends on. That is
prod_{a in S} (q_a - 1) free parameters, so

    dim G_d = sum_{|S| <= d} prod_{a in S} (q_a - 1),

counting the constant. The indicator parameterization a ridge is fitted in is larger -- it
carries one weight per level and per level pair, and the penalty picks one representative of
each equivalence class -- so the two counts are both reported: the indicator count is what the
design matrix has, the dimension is what has to be determined.

Identifiability is the separate, combinatorial condition. Least squares determines a degree-2
model only if the design visits every pair of levels of every pair of factors at least once;
no number of labels concentrated on a subset of the pair cells will do it, and a design can
cover a vanishing fraction of the grid and still cover every cell. That is what happens on the
FractalNet record, and the covered/total counts come from s27_signal_anatomy.py rather than
being recomputed here, so the two scripts cannot drift apart.

Outputs
-------
results/dimension.json      the counts behind the corollary

    python3 analysis/s66_dimension.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')

import numpy as np                                                        # noqa: E402

import surrogate                                                          # noqa: E402


def dim_le(card, d):
    """dim G_d = sum over subsets of size <= d of prod (q_a - 1), the constant included."""
    card = list(card)
    m = len(card)
    # one running total per subset size, built by the standard elementary-symmetric recursion
    e = [0.0] * (m + 1)
    e[0] = 1.0
    for q in card:
        for k in range(m, 0, -1):
            e[k] = e[k] + e[k - 1] * (q - 1)
    return int(round(sum(e[:d + 1])))


def indicator_features(card, pairs=True):
    """What the fitted design matrix actually carries: one column per level and level pair."""
    card = list(card)
    n = int(sum(card))
    if pairs:
        n += int(sum(card[i] * card[j] for i in range(len(card))
                     for j in range(i + 1, len(card))))
    return n


def pair_cells(card):
    """Pair cells: the combinatorial object a degree-2 design has to cover."""
    card = list(card)
    return int(sum(card[i] * card[j] for i in range(len(card))
                   for j in range(i + 1, len(card))))


def grid(card):
    out = 1
    for q in card:
        out *= int(q)
    return out


def describe(name, card, note):
    card = [int(q) for q in card]
    g = grid(card)
    row = {
        'name': name, 'cardinalities': card, 'n_factors': len(card),
        'grid_size': g, 'grid_size_log10': math.log10(g), 'note': note,
        'pair_cells': pair_cells(card),
        'indicator_features_degree2': indicator_features(card),
        'dim': {str(d): dim_le(card, d) for d in range(1, len(card) + 1)},
    }
    row['dim_degree2'] = row['dim']['2']
    row['grid_over_dim2'] = g / row['dim_degree2']
    return row


def main():
    rows = [
        describe('NAS-Bench-201', [5] * 6,
                 'six edges, five operations each; the complete factorial Section IV '
                 'decomposes'),
        describe('NAS-Bench-Macro', [3] * 8,
                 'eight layers, three blocks each'),
        describe('FractalNet record', surrogate.CARD,
                 'the 13 genes of the released per-architecture record'),
    ]
    out = {'rows': {r['name']: r for r in rows},
           'formula': 'dim G_d = sum_{|S|<=d} prod_{a in S} (q_a - 1), the constant included',
           'identifiability': 'least squares determines a degree-2 model only if the design '
                              'covers every pair cell; coverage, not count, is the condition'}

    #: the FractalNet coverage statement of the corollary, taken from the script that owns it
    try:
        with open(os.path.join(RES, 'signal_anatomy.json')) as fh:
            sig = json.load(fh)
    except FileNotFoundError:
        out['coverage'] = None
    else:
        out['coverage'] = {
            ds: {'pair_cells': sig[ds]['identifiability']['pair_cells'],
                 'pair_cells_seen': sig[ds]['identifiability']['pair_cells_seen'],
                 'grid_fraction': sig[ds]['identifiability']['grid_fraction'],
                 'labels_per_cell': sig[ds]['identifiability']['labels_per_cell']}
            for ds in ('cifar10', 'cifar100') if ds in sig}
        fr = out['rows']['FractalNet record']
        out['checks_coverage'] = {
            'pair_cell_count_matches_s27': all(
                v['pair_cells'] == fr['pair_cells'] for v in out['coverage'].values()),
            'every_pair_cell_covered': all(
                v['pair_cells_seen'] == v['pair_cells'] for v in out['coverage'].values()),
        }

    nb = out['rows']['NAS-Bench-201']
    out['checks'] = {
        #: the degree-6 dimension of a complete 5^6 grid is the grid itself: the decomposition
        #: is a change of basis and loses nothing, which is the sanity check on the formula
        'full_degree_dimension_is_the_grid': dim_le([5] * 6, 6) == 5 ** 6,
        'macro_full_degree_dimension_is_the_grid': dim_le([3] * 8, 8) == 3 ** 8,
        'dimension_below_indicator_count': all(
            r['dim_degree2'] <= r['indicator_features_degree2'] + 1 for r in rows),
    }

    os.makedirs(RES, exist_ok=True)
    jp = os.path.join(RES, 'dimension.json')
    with open(jp, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    report(out)
    print('\nwrote %s' % os.path.relpath(jp, ROOT))
    return 0


def report(o):
    print('what a degree-2 model has to determine, against the space it lives on\n')
    print('  %-24s %14s %10s %10s %10s'
          % ('grid', 'architectures', 'dim G_2', 'features', 'pair cells'))
    for r in o['rows'].values():
        print('  %-24s %14s %10d %10d %10d'
              % (r['name'],
                 ('%.0e' % r['grid_size']) if r['grid_size'] >= 1e7
                 else '{:,}'.format(r['grid_size']),
                 r['dim_degree2'], r['indicator_features_degree2'], r['pair_cells']))
    if o.get('coverage'):
        for ds, v in o['coverage'].items():
            print('  FractalNet/%-13s %s of %s pair cells covered from %.3f%% of the grid, '
                  'at %.1f labels each'
                  % (ds, '{:,}'.format(v['pair_cells_seen']),
                     '{:,}'.format(v['pair_cells']), 100 * v['grid_fraction'],
                     v['labels_per_cell']))
    c = dict(o['checks'])
    c.update(o.get('checks_coverage', {}))
    print('\nchecks: ' + ' | '.join('%s: %s' % (k.replace('_', ' '), v) for k, v in c.items()))


if __name__ == '__main__':
    sys.exit(main())
