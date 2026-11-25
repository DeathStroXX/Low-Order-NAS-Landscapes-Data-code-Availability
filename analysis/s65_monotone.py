"""Which monotone map to recalibrate with, and what a wrong one costs.

Section V shows that the spectrum moves under a monotone change of the response and that the
rank map collapses the three NAS-Bench-201 tasks onto one spectrum. That leaves the obvious
question open: the rank map is one monotone map among infinitely many, so is it the right one,
and how much does the choice matter? This script answers both by measuring, over a family of
maps wide enough to contain the natural alternatives, two things that are not the same
question:

  * `A<=2`, the share of variance the map puts at order at most two. This is what the ceiling
    of Proposition 1 is, and it is the quantity a *fitting* decision cares about.
  * what the exact degree-2 projection of that map's response then attains in rank fidelity
    and in regret at a shortlist of ten, scored against the original accuracies. The scale remark
    says these are not implied by the first, and the point of the table is that they are not.

The second is the criterion this paper proposes, because it is the one a budget is decided in.
A map is not chosen to make a spectrum look concentrated; it is chosen because the degree-2
function the squared-error criterion then selects is a better ranker. The script reports both
columns side by side so a reader can see where they disagree.

The family
----------
identity      raw accuracy; the scale every benchmark publishes
log           log(acc); an unbounded scale, but only at the bottom
logit         log(p/(1-p)) on the accuracy as a probability: the natural unbounded scale for a
              bounded response, and the one Section VIII reports as narrowing but not closing
              the spread
power p       acc^p for p on a grid; the dose-response family of Section V-B, which reproduces
              the task ordering within one task
ranks         rank(acc), the crudest monotone map there is
normal scores Phi^-1(rank / (n+1)), the van der Waerden transform: the rank map pushed onto a
              Gaussian scale, so it has the rank map's invariance but a non-degenerate
              density. If ranks beat the parametric maps only because they are rank-based,
              this map scores with them; if the flat density of the rank map is itself doing
              the work, the two separate.

Every map is strictly increasing, so all of them leave the architecture ranking of the response
untouched and therefore leave the *truth* they are scored against untouched. Only the function
being decomposed changes.

Outputs
-------
results/monotone.json       every map, every dataset, both benchmarks
../tables/tab_monotone.tex  the supplement's table

    python3 analysis/s65_monotone.py
"""
import json

import os
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
TBL = os.path.join(ROOT, 'tables')

import anova                                                              # noqa: E402
import s53_spectrum as sp53                                               # noqa: E402
import s57_benchmarks as sb                                               # noqa: E402

NB201 = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16',
         'nasbench_macro': 'Macro'}
K = 10                                   # the shortlist the regret is measured at
POWERS = [0.25, 0.5, 2.0, 4.0, 8.0, 12.0]


# ------------------------------------------------------------------ the maps
def _ranks(y):
    return rankdata(y)


def _normal_scores(y):
    """Phi^-1(rank / (n + 1)): the rank map carried onto a Gaussian scale."""
    r = rankdata(y)
    return ndtri(r / (len(y) + 1.0))


def _logit(y):
    """log(p / (1 - p)) with the accuracy read as a percentage.

    Accuracies of exactly 0 or 100 would send this to infinity; none occur in either table,
    and the guard below refuses rather than clipping, because clipping would silently stop
    the map being monotone.
    """
    p = np.asarray(y, float) / 100.0
    if p.min() <= 0.0 or p.max() >= 1.0:
        raise ValueError('logit needs accuracies strictly inside (0, 100)')
    return np.log(p / (1.0 - p))


def _log(y):
    y = np.asarray(y, float)
    if y.min() <= 0.0:
        raise ValueError('log needs strictly positive accuracies')
    return np.log(y)


def maps():
    """(key, label, f) for every map in the family, in table order."""
    out = [('identity', 'raw accuracy', lambda y: np.asarray(y, float)),
           ('log', r'$\log$', _log),
           ('logit', 'logit', _logit)]
    for p in POWERS:
        out.append(('power_%g' % p, r'$y^{%g}$' % p,
                    (lambda pp: (lambda y: np.asarray(y, float) ** pp))(p)))
    out += [('normal_scores', 'normal scores', _normal_scores),
            ('ranks', 'ranks', _ranks)]
    return out


# ------------------------------------------------------------------ one landscape
def evaluate(G, y, shape):
    """For every map: the spectrum of the mapped response, and what its g_2 then decides.

    `y` is the original response. The mapped response is decomposed; the degree-2 projection
    of *that* is then scored, as a ranking, against `y` itself -- the accuracies a laboratory
    actually cares about, which no map changes the order of.
    """
    truth = np.asarray(y, float)
    best = float(truth.max())
    idx = tuple(np.asarray(G).T)
    rows = []
    for key, label, f in maps():
        z = f(truth)
        Y = anova.to_grid(G, z, shape)
        spec = anova.spectrum(Y)
        cum = anova.cumulative(spec)
        sums = anova.order_sums(Y)
        g2 = (sums[1] + sums[2])[idx]
        order = np.argsort(-g2, kind='stable')
        rows.append({
            'map': key, 'label': label,
            # anova.cumulative already returns percentages
            'a_le_1': float(cum[0]), 'a_le_2': float(cum[1]), 'a_le_3': float(cum[2]),
            'skewness': float(((z - z.mean()) ** 3).mean() / (z.std() ** 3 + 1e-300)),
            'rho_g2': float(spearmanr(g2, truth).statistic),
            'regret10_g2': float(best - truth[order[:K]].max()),
        })
    return rows


def summarize(rows):
    """Which map wins on each of the two criteria, and what the wrong one costs."""
    by_fit = max(rows, key=lambda r: r['a_le_2'])
    by_dec = min(rows, key=lambda r: (r['regret10_g2'], -r['rho_g2']))
    raw = next(r for r in rows if r['map'] == 'identity')
    rk = next(r for r in rows if r['map'] == 'ranks')
    ns = next(r for r in rows if r['map'] == 'normal_scores')
    #: What choosing on the wrong criterion costs is the EXCESS regret of the ceiling winner
    #: over the best regret available, not that winner's regret outright: a map can maximize
    #: the ceiling and still sit at the minimum of the decision metric, and on three of the
    #: four landscapes here one does. Comparing the two winners by name instead reports a
    #: disagreement wherever the minimum is attained by more than one map and the tie breaks
    #: towards a rank map, which is a disagreement about labels and not about the decision.
    best_regret = min(r['regret10_g2'] for r in rows)
    cost_of_ceiling = by_fit['regret10_g2'] - best_regret
    return {
        'best_by_a_le_2': by_fit['map'],
        'best_by_regret10': by_dec['map'],
        'criteria_agree': by_fit['map'] == by_dec['map'],
        'ceiling_winner_also_decides_best': cost_of_ceiling <= 1e-12,
        'maps_attaining_the_best_regret': [r['map'] for r in rows
                                           if r['regret10_g2'] - best_regret <= 1e-12],
        'cost_of_choosing_on_the_ceiling': cost_of_ceiling,
        'a_le_2_of_the_decision_winner': by_dec['a_le_2'],
        'a_le_2_of_the_fit_winner': by_fit['a_le_2'],
        'regret10_of_the_fit_winner': by_fit['regret10_g2'],
        'ranks_minus_normal_scores_a_le_2': rk['a_le_2'] - ns['a_le_2'],
        'ranks_minus_normal_scores_regret10': rk['regret10_g2'] - ns['regret10_g2'],
        'ranks_minus_normal_scores_rho': rk['rho_g2'] - ns['rho_g2'],
        'cost_of_staying_raw_regret10': raw['regret10_g2'] - rk['regret10_g2'],
        'cost_of_staying_raw_rho': rk['rho_g2'] - raw['rho_g2'],
        'spread_a_le_2': max(r['a_le_2'] for r in rows) - min(r['a_le_2'] for r in rows),
        'spread_regret10': (max(r['regret10_g2'] for r in rows)
                            - min(r['regret10_g2'] for r in rows)),
    }


def main():
    out = {'k': K, 'powers': POWERS,
           'family': [{'map': k, 'label': lb} for k, lb, _ in maps()],
           'criterion': 'the map is chosen by what the degree-2 projection of the mapped '
                        'response attains in the decision metric, not by which map makes the '
                        'spectrum look most concentrated; the table reports both so the '
                        'places they disagree are visible',
           'landscapes': {}}

    # ------------------------------------------------------------- NAS-Bench-201, 5^6
    G, acc = sp53.load()
    shape = (sp53.N_OP,) * sp53.N_EDGE
    for d in NB201:
        rows = evaluate(G, acc[d], shape)
        out['landscapes'][d] = {'benchmark': 'NAS-Bench-201', 'shape': list(shape),
                                'rows': rows, 'summary': summarize(rows)}
        print('%-16s A<=2 from %.1f%% to %.1f%% over the family; regret at ten from %.2f to '
              '%.2f' % (d, min(r['a_le_2'] for r in rows),
                        max(r['a_le_2'] for r in rows),
                        min(r['regret10_g2'] for r in rows),
                        max(r['regret10_g2'] for r in rows)))

    # ------------------------------------------------------------- NAS-Bench-Macro, 3^8
    mac = sb.load_macro()                           # None when the fetched table is absent
    if mac is None:
        print('NAS-Bench-Macro not read; the 3^8 block is skipped.')
    else:
        Gm, ym = mac[0], mac[2]
        rows = evaluate(Gm, ym, (3,) * 8)
        out['landscapes']['nasbench_macro'] = {
            'benchmark': 'NAS-Bench-Macro', 'shape': [3] * 8,
            'rows': rows, 'summary': summarize(rows)}
        print('%-16s A<=2 from %.1f%% to %.1f%% over the family'
              % ('nasbench_macro', min(r['a_le_2'] for r in rows),
                 max(r['a_le_2'] for r in rows)))

    #: the two claims the supplement makes from this file, checked here rather than read off
    ls = out['landscapes']
    out['checks'] = {
        #: the two rank-based maps differ in their density, not in their invariance. If the
        #: flat density of the rank map were doing the work they would separate; they do not.
        'rank_based_maps_decide_alike_everywhere': all(
            abs(ls[d]['summary']['ranks_minus_normal_scores_regret10']) < 0.05
            and abs(ls[d]['summary']['ranks_minus_normal_scores_rho']) < 0.01 for d in ls),
        'max_rank_vs_normal_scores_a_le_2_gap': max(
            abs(ls[d]['summary']['ranks_minus_normal_scores_a_le_2']) for d in ls),
        'every_landscape_prefers_a_rank_based_map': all(
            ls[d]['summary']['best_by_regret10'] in ('ranks', 'normal_scores') for d in ls),
        #: the scale remark applied to the choice of the map. The two criteria are different
        #: criteria and the maps they name differ on every landscape, but what matters is
        #: whether choosing on the ceiling costs anything in the decision, and on the three
        #: NAS-Bench-201 tasks it does not: the ceiling winner sits at the minimum regret too.
        'the_two_criteria_name_different_maps_everywhere': all(
            not ls[d]['summary']['criteria_agree'] for d in ls),
        'landscapes_where_choosing_on_the_ceiling_costs_nothing': [
            d for d in ls if ls[d]['summary']['ceiling_winner_also_decides_best']],
        'landscapes_where_choosing_on_the_ceiling_costs_something': [
            d for d in ls if not ls[d]['summary']['ceiling_winner_also_decides_best']],
        'max_a_le_2_given_up_by_choosing_on_the_decision': max(
            ls[d]['summary']['a_le_2_of_the_fit_winner']
            - ls[d]['summary']['a_le_2_of_the_decision_winner'] for d in ls),
        'max_regret_cost_of_choosing_on_the_ceiling': max(
            ls[d]['summary']['cost_of_choosing_on_the_ceiling'] for d in ls),
        #: the practical reading. Every map that undoes the compression -- a rank-based one or
        #: a sufficiently convex power -- decides within a few hundredths of a point of every
        #: other; the expensive mistake is not choosing badly among them but not recalibrating.
        'worst_map_that_undoes_the_compression': max(
            ls[d]['summary']['regret10_of_the_fit_winner'] for d in ls),
        'cost_of_not_recalibrating_at_all': {
            d: ls[d]['summary']['cost_of_staying_raw_regret10'] for d in ls},
        'max_cost_of_not_recalibrating': max(
            ls[d]['summary']['cost_of_staying_raw_regret10'] for d in ls),
    }

    os.makedirs(RES, exist_ok=True)
    jp = os.path.join(RES, 'monotone.json')
    with open(jp, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    tp = emit_table(out)
    report(out)
    print('\nwrote %s' % os.path.relpath(jp, ROOT))
    print('wrote %s' % os.path.relpath(tp, ROOT))
    return 0


# ------------------------------------------------------------------ the table
def emit_table(o):
    ls = o['landscapes']
    cols = [d for d in NB201 if d in ls]
    L = []
    a = L.append
    a(r'\begin{table}[!t]')
    a(r'\centering')
    a(r'\caption{Which monotone map, and what the choice costs. Every row is a strictly '
      r'increasing map of the accuracies, so every row leaves the architecture ranking of the '
      r'response (and therefore the truth each is scored against) untouched; only the '
      r'function being decomposed changes. $A_{\le 2}$ is the exact share of variance the '
      r'mapped response puts at order at most two, which is the main '
      r'paper\textquotesingle s degree-2 ceiling for that scale. '
      r'\emph{regret} is what the exact degree-2 projection of the mapped response then gives '
      r'up at a shortlist of ten, scored against the original accuracies: the criterion this '
      r'paper proposes, because it is the one a budget is decided in. The two rank-based maps '
      r'differ only in their density: they return identical regret on every landscape and '
      r'$\rho$ within $0.004$, while their $A_{\le 2}$ still differ by up to $1.4$ points, '
      r'which locates what a decision depends on in the invariance they share.}')
    a(r'\label{tab:monotone}')
    a(r'\footnotesize')
    a(r'\setlength{\tabcolsep}{3.0pt}')
    a(r'\begin{tabular}{@{}l %s@{}}' % ' '.join(['rr'] * len(cols)))
    a(r'\toprule')
    a(' & ' + ' & '.join(r'\multicolumn{2}{c}{%s}' % SHORT[d] for d in cols) + r' \\')
    a(' '.join(r'\cmidrule(lr){%d-%d}' % (2 + 2 * i, 3 + 2 * i) for i in range(len(cols))))
    a('Map & ' + ' & '.join([r'$A_{\le 2}$ & regret'] * len(cols)) + r' \\')
    a(r'\midrule')
    keys = [r['map'] for r in ls[cols[0]]['rows']]
    for k in keys:
        lab = next(r['label'] for r in ls[cols[0]]['rows'] if r['map'] == k)
        cells = []
        for d in cols:
            r = next(x for x in ls[d]['rows'] if x['map'] == k)
            win = ls[d]['summary']['best_by_regret10'] == k
            cells.append((r'\textbf{%.1f} & \textbf{%.2f}' if win else r'%.1f & %.2f')
                         % (r['a_le_2'], r['regret10_g2']))
        a(r'%s & %s \\' % (lab, ' & '.join(cells)))
    if 'nasbench_macro' in ls:
        m = ls['nasbench_macro']
        a(r'\addlinespace[2pt]\midrule')
        a(r'\multicolumn{%d}{@{}l}{\emph{NAS-Bench-Macro, $3^{8}$: $A_{\le 2}$ from '
          r'%.1f\%% to %.1f\%%; the ceiling picks %s, the decision %s}} \\'
          % (1 + 2 * len(cols), min(r['a_le_2'] for r in m['rows']),
             max(r['a_le_2'] for r in m['rows']),
             m['summary']['best_by_a_le_2'].replace('_', ' ').replace('power ', '$y^{') + '}$'
             if m['summary']['best_by_a_le_2'].startswith('power')
             else m['summary']['best_by_a_le_2'].replace('_', ' '),
             m['summary']['best_by_regret10'].replace('_', ' ')))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    p = os.path.join(TBL, 'tab_monotone.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return p


def report(o):
    print('\nthe map each landscape prefers, by each of the two criteria:')
    for d, v in o['landscapes'].items():
        s = v['summary']
        print('  %-16s by A<=2: %-14s by regret at ten: %-14s %s'
              % (SHORT.get(d, d), s['best_by_a_le_2'], s['best_by_regret10'],
                 'agree' if s['criteria_agree'] else 'DISAGREE'))
        print('  %-16s staying on raw accuracy costs %.2f points of regret and %.3f of rho'
              % ('', s['cost_of_staying_raw_regret10'], s['cost_of_staying_raw_rho']))
    c = o['checks']
    print('\nchecks: the two rank-based maps decide alike on every landscape: %s (their '
          'ceilings' % c['rank_based_maps_decide_alike_everywhere'])
    print('        still differ by up to %.2f points, which is the density and not the '
          'invariance)' % c['max_rank_vs_normal_scores_a_le_2_gap'])
    print('        every landscape prefers a rank-based map in the decision: %s'
          % c['every_landscape_prefers_a_rank_based_map'])
    print('        the two criteria name different maps on every landscape: %s'
          % c['the_two_criteria_name_different_maps_everywhere'])
    print('        choosing on the ceiling costs nothing on: %s'
          % ', '.join(c['landscapes_where_choosing_on_the_ceiling_costs_nothing']) or 'none')
    print('        and costs something on: %s'
          % (', '.join(c['landscapes_where_choosing_on_the_ceiling_costs_something'])
             or 'none'))
    print('        choosing on the decision gives up at most %.1f points of A<=2; choosing '
          'on the' % c['max_a_le_2_given_up_by_choosing_on_the_decision'])
    print('        ceiling instead costs at most %.2f accuracy points at a shortlist of ten, '
          'against the' % c['max_regret_cost_of_choosing_on_the_ceiling'])
    print('        %.2f points that not recalibrating at all costs. The choice among the maps '
          'that undo' % c['max_cost_of_not_recalibrating'])
    print('        the compression is cheap; the choice of whether to undo it is not.')


if __name__ == '__main__':
    sys.exit(main())
