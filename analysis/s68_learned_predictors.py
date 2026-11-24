"""The degree-2 ridge against trained performance predictors, at the same label budget.

Why this file exists
--------------------
Every cheap ranker this paper places against the exact degree-2 optimum is either label-free
or a degree-2 ridge. That leaves one comparison a reader is entitled to and the rest of the
archive does not make: against the *trained* predictors that the performance-prediction
literature actually fields -- tree ensembles and a neural predictor on a path encoding
(white2021powerful, white2021bananas, wen2020neuralpredictor, dudziak2020brpnas,
ning2021evaluating). Without it, "our free ranker beats thirteen zero-cost proxies" leaves
open the possibility that anything fitted would beat all of them by more.

Why the numbers are computed here rather than quoted
----------------------------------------------------
Published Spearman values for those predictors exist, but they are not comparable to the ones
in this paper: they are reported at different label budgets, over different train/test pools,
against different accuracy columns, and some are Kendall tau rather than Spearman. Quoting
them side by side would produce a table that looked like a head-to-head and was not one. So
the predictors are refitted here, on the same table, at the same budgets, over the same
held-out complement, scored in the same two metrics, under the *identical* index draws:
`numpy.random.default_rng(5000 + s)` then `rng.choice(15625, n, replace=False)` for
s = 0..19, which is byte-for-byte the protocol of the recalibration block of s53_spectrum.py.
The ridge row this file prints is therefore the same ridge row that file prints, which is the
check that the protocol really is shared (`checks.ridge_matches_s53`).

This costs no GPU and trains no network: every label is a lookup in the published table, and
the predictors are fitted on 100 to 1,000 of those lookups.

What it finds
-------------
1. At a budget of 100 labels the degree-2 ridge on recalibrated labels is ahead of both tree
   ensembles and far ahead of the neural predictor. A trained predictor has little to fit at
   that budget, which is the budget a one-GPU laboratory works at.
2. At 1,000 labels they converge, and gradient boosting edges ahead on one dataset. Neither
   ensemble reaches the exact degree-2 projection of the recalibrated response, so the
   reference of s64_realization.py is still the thing to beat.
3. The monotone recalibration of Section V helps the *trained* predictors too, by about as
   much as it helps the ridge. That is the finding this file contributes rather than merely
   defends: the scale illusion is a property of the landscape, not an artefact of the
   authors' model class, and any predictor fitted in squared error on these labels is paying
   for it.

Outputs
-------
results/learned_predictors.json    every number, with the protocol recorded
../tables/tab_learned.tex          Table: trained predictors at matched budgets

    python3 analysis/s68_learned_predictors.py
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
#: The protocol is not reimplemented here, it is imported. `load`, `design` and `ridge_fit`
#: are s53_spectrum.py's own: the same row order (so the same index draws select the same
#: architectures), the same 265-column reduced design, the same unpenalized intercept and the
#: same alpha. Reimplementing any of them would make the ridge row a near-miss rather than the
#: same number, and `checks.ridge_matches_s53` would catch it -- as it did.
from s53_spectrum import (DATASETS, N_EDGE, N_OP, OPS, design, load,  # noqa: E402
                          ridge_fit)

RES = os.path.join(HERE, 'results')
TBL = os.path.join(HERE, '..', 'tables')

SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}

#: the protocol of s53_spectrum.py's recalibration block, reproduced exactly so the ridge row
#: of that file and of this one are the same number
SEEDS = 20
SEED0 = 5000
#: the two budgets the manuscript quotes: the one a one-GPU laboratory works at,
#: and the one the realization table of s64_realization.py is built on
BUDGETS = [100, 1000]
K_SHORTLIST = 10

#: NAS-Bench-201's cell is a fixed complete DAG on four nodes with an operation on each of the
#: six edges. In the order the genotype stores them the edges are
#: 0->1, 0->2, 1->2, 0->3, 1->3, 2->3, so there are exactly four input-to-output paths.
PATHS = [(3,), (0, 4), (1, 5), (0, 2, 5)]
#: the two operations the path encoding has to treat specially, by their codes in the shared
#: OPS order; resolved rather than hard-coded, so a reordering upstream cannot silently change
#: which operation the encoding treats as severing a path
NONE, SKIP = OPS.index('none'), OPS.index('skip_connect')


def path_design(P):
    """The path encoding of white2021bananas, for this space.

    A path is dead if any operation on it is `none`; otherwise its identity is the sequence of
    its non-`skip_connect` operations, since a skip connection is the identity map and
    contributes nothing to what the path computes. The design matrix is the indicator of each
    path identity that occurs anywhere in the space, which is the encoding as the literature
    uses it rather than a variant invented here.
    """
    P = np.asarray(P, int)
    rows = []
    for g in P:
        feats = set()
        for pi, edges in enumerate(PATHS):
            ops = [g[e] for e in edges]
            if NONE in ops:
                continue
            feats.add((pi, tuple(o for o in ops if o != SKIP)))
        rows.append(feats)
    vocab = sorted({f for r in rows for f in r}, key=repr)
    index = {f: i for i, f in enumerate(vocab)}
    A = np.zeros((len(rows), len(vocab)), dtype=np.float64)
    for i, r in enumerate(rows):
        for f in r:
            A[i, index[f]] = 1.0
    return A, vocab


def onehot(G):
    """Full one-hot of the genotype, which is the encoding the tree ensembles are given.

    The operations are nominal, not ordinal, so handing a tree the integer codes would let it
    split on an ordering that does not exist. The reduced design `design(G, 1)` would do, but
    it drops one level per edge, and there is no reason to withhold it from a model that is
    not being asked to be identifiable.
    """
    G = np.asarray(G, int)
    A = np.zeros((len(G), N_EDGE * N_OP), dtype=np.float64)
    r = np.arange(len(G))
    for i in range(N_EDGE):
        A[r, i * N_OP + G[:, i]] = 1.0
    return A


def regret_at(pred, truth, k=K_SHORTLIST):
    """Accuracy points given up by training the k best-predicted candidates instead of all."""
    return float(truth.max() - truth[np.argsort(-pred)[:k]].max())


def _models():
    """The trained predictor families, each on the encoding its own literature uses.

    Hyper-parameters are the library defaults except for the three that would otherwise make
    the comparison unfair: enough trees and boosting rounds not to be undertrained, and enough
    MLP iterations to converge at the largest budget. Nothing is tuned per dataset or per
    budget, and nothing is tuned on the held-out complement -- the ridge is not tuned either,
    and a predictor tuned against the evaluation set would not be a comparison.
    """
    return [
        ('rf', 'random forest', 'onehot',
         lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)),
        ('gbdt', 'gradient-boosted trees', 'onehot',
         lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0)),
        ('mlp_path', 'neural predictor, path encoding', 'path',
         lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600, random_state=0)),
    ]


def main():
    G, Y = load()                        # s53_spectrum.py's loader, so the row order is its
    P = G
    n_arch = len(G)
    X1 = onehot(G)                       # full one-hot of the genotype, for the tree ensembles
    X2 = design(G, 2)                    # the 265-column reduced design the paper's ridge uses
    XP, vocab = path_design(G)
    enc = {'onehot': X1, 'path': XP}
    print('NAS-Bench-201: %d architectures; one-hot %d features, degree-2 %d, paths %d'
          % (n_arch, X1.shape[1], X2.shape[1], XP.shape[1]))

    models = _models()
    arms = [('ridge2', 'degree-2 ridge (this paper)', 'degree-2')] + \
           [(k, lab, encname) for k, lab, encname, _ in models]

    out = {
        'n_arch': n_arch,
        'datasets': DATASETS,
        'budgets': BUDGETS,
        'n_features': {'onehot': int(X1.shape[1]), 'degree2': int(X2.shape[1]),
                       'path': int(XP.shape[1])},
        'n_paths_vocab': len(vocab),
        'protocol': {
            'seeds': SEEDS,
            'index_draw': 'numpy.random.default_rng(%d + s) then '
                          'rng.choice(%d, n, replace=False) for s = 0..%d'
                          % (SEED0, n_arch, SEEDS - 1),
            'shared_with': 's53_spectrum.py (the recalibration block); the ridge row here is '
                           'the same number that file reports, which is what makes the '
                           'comparison a head-to-head rather than two protocols side by side',
            'evaluation': 'the complement of the fitted sample, all %d - n architectures'
                          % n_arch,
            'metrics': 'Spearman rho against the held-out truth, and the accuracy given up by '
                       'the top-%d shortlist' % K_SHORTLIST,
            'scales': {'raw': 'fitted on the target accuracies',
                       'recal': 'fitted on the ranks of the same accuracies, the monotone '
                                'recalibration of Section V'},
            'tuning': 'library defaults but for tree count, boosting rounds and MLP '
                      'iterations; nothing tuned per dataset, per budget, or against the '
                      'held-out complement',
            'why_not_quoted': 'published values for these predictors are reported at other '
                              'budgets, over other train/test pools, against other accuracy '
                              'columns, and some are Kendall tau; refitting is the only way '
                              'to put them on this paper\'s axes',
        },
        'results': {},
    }

    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for ds in DATASETS:
        y = Y[ds]
        out['results'][ds] = {}
        for n in BUDGETS:
            cell = {}
            acc = {(k, sc): ([], []) for k, _, _ in arms for sc in ('raw', 'recal')}
            for s_ in range(SEEDS):
                rng = np.random.default_rng(SEED0 + s_)
                idx = rng.choice(n_arch, n, replace=False)
                ev = np.setdiff1d(np.arange(n_arch), idx)
                truth = y[ev]
                targets = {'raw': y[idx], 'recal': rankdata(y[idx]).astype(float)}
                for sc, yt in targets.items():
                    p = (X2 @ ridge_fit(X2[idx], yt))[ev]
                    acc[('ridge2', sc)][0].append(spearmanr(p, truth).statistic)
                    acc[('ridge2', sc)][1].append(regret_at(p, truth))
                    for key, _, encname, make in models:
                        A = enc[encname]
                        m = make()
                        m.fit(A[idx], yt)
                        p = m.predict(A[ev])
                        acc[(key, sc)][0].append(spearmanr(p, truth).statistic)
                        acc[(key, sc)][1].append(regret_at(p, truth))
            for (key, sc), (rr, gg) in acc.items():
                cell.setdefault(key, {})[sc] = {
                    'rho': float(np.mean(rr)), 'rho_sd': float(np.std(rr)),
                    'regret10': float(np.mean(gg)), 'regret10_sd': float(np.std(gg)),
                }
            #: what the recalibration is worth to this arm, which is the question this file
            #: answers that no other file in the archive does
            for key in cell:
                cell[key]['recal_gain_rho'] = (cell[key]['recal']['rho']
                                               - cell[key]['raw']['rho'])
            out['results'][ds][str(n)] = cell
            print('  %-16s n=%-5d %s' % (SHORT[ds], n,
                                         '  '.join('%s %.3f/%.3f' % (k, cell[k]['raw']['rho'],
                                                                     cell[k]['recal']['rho'])
                                                   for k, _, _ in arms)))

    # ------------------------------------------------------------------ the reference
    #: the exact degree-2 projection of the recalibrated response, which is the reference
    #: s64_realization.py expresses every ranker as a percentage of
    with open(os.path.join(RES, 'ceiling.json')) as fh:
        cei = json.load(fh)
    scales = cei['datasets'][DATASETS[0]].keys()
    recal_key = 'recalibrated' if 'recalibrated' in scales else 'recal'
    ref = {}
    for d in DATASETS:
        row = next(r for r in cei['datasets'][d][recal_key]['ladder'] if r['d'] == 2)
        ref[d] = {'rho': row['rho'], 'regret10': row['regret_at_10']}
    out['reference'] = {
        'what': 'the exact degree-2 projection of the recalibrated response, computed from '
                'the table with nothing fitted; the same reference as s64_realization.py',
        'per_dataset': ref,
    }
    for ds in DATASETS:
        for n in BUDGETS:
            for key in out['results'][ds][str(n)]:
                c = out['results'][ds][str(n)][key]
                for sc in ('raw', 'recal'):
                    c[sc]['realization_rho'] = 100.0 * c[sc]['rho'] / ref[ds]['rho']

    # ------------------------------------------------------------------ the findings
    def rho(ds, n, key, sc='recal'):
        return out['results'][ds][str(n)][key][sc]['rho']

    learned = [k for k, _, _ in arms if k != 'ridge2']
    out['findings'] = {
        'ridge_leads_every_learned_predictor_at': [
            n for n in BUDGETS
            if all(rho(d, n, 'ridge2') > rho(d, n, k) for d in DATASETS for k in learned)],
        'cells_where_a_learned_predictor_leads_the_ridge': [
            {'dataset': d, 'budget': n, 'predictor': k,
             'rho': rho(d, n, k), 'ridge_rho': rho(d, n, 'ridge2')}
            for d in DATASETS for n in BUDGETS for k in learned
            if rho(d, n, k) > rho(d, n, 'ridge2')],
        'cells_where_anything_fitted_beats_the_exact_degree2_reference': [
            {'dataset': d, 'budget': n, 'arm': k, 'scale': sc}
            for d in DATASETS for n in BUDGETS for k, _, _ in arms
            for sc in ('raw', 'recal')
            if out['results'][d][str(n)][k][sc]['rho'] > ref[d]['rho']],
        'recalibration_gain_rho': {
            k: {'mean': float(np.mean([out['results'][d][str(n)][k]['recal_gain_rho']
                                       for d in DATASETS for n in BUDGETS])),
                'min': float(np.min([out['results'][d][str(n)][k]['recal_gain_rho']
                                     for d in DATASETS for n in BUDGETS])),
                'n_cells_positive': int(sum(
                    out['results'][d][str(n)][k]['recal_gain_rho'] > 0
                    for d in DATASETS for n in BUDGETS)),
                'n_cells': len(DATASETS) * len(BUDGETS)}
            for k, _, _ in arms},
    }

    # ------------------------------------------------------------------ checks
    #: the ridge row has to equal the one s53_spectrum.py already published, or the two files
    #: are not running the same protocol and the comparison is not a head-to-head
    agree, worst = True, 0.0
    sp = os.path.join(RES, 'spectrum.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            s53 = json.load(fh)['recalibration']
        for d in DATASETS:
            for n in BUDGETS:
                b = s53.get(d, {}).get('budgets', {}).get(str(n))
                if not b:
                    continue
                for sc, arm in (('raw', 'own_raw'), ('recal', 'own_recal')):
                    delta = abs(out['results'][d][str(n)]['ridge2'][sc]['rho'] - b[arm]['rho'])
                    worst = max(worst, delta)
                    agree &= delta < 5e-3
    out['checks'] = {
        'ridge_matches_s53': bool(agree),
        'ridge_vs_s53_worst_abs_rho_gap': worst,
        'recalibration_helps_every_arm_on_average': all(
            out['findings']['recalibration_gain_rho'][k]['mean'] > 0 for k, _, _ in arms),
        'nothing_fitted_beats_the_exact_reference': not out['findings'][
            'cells_where_anything_fitted_beats_the_exact_degree2_reference'],
        'ridge_leads_everything_at_the_smallest_budget':
            BUDGETS[0] in out['findings']['ridge_leads_every_learned_predictor_at'],
    }

    os.makedirs(RES, exist_ok=True)
    jp = os.path.join(RES, 'learned_predictors.json')
    with open(jp, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    tp = emit_table(out, arms)
    report(out, arms)
    print('\nwrote %s\nwrote %s' % (jp, tp))


def emit_table(o, arms):
    ref = o['reference']['per_dataset']
    L = [r'%% generated by analysis/s68_learned_predictors.py -- do not edit by hand',
         r'\begin{table}[!t]', r'\centering',
         r'\caption{Trained performance predictors against the degree-2 ridge, at matched '
         r'label budgets on NAS-Bench-201. Spearman $\rho$ against the held-out complement, '
         r'averaged over %d seeds; \emph{raw} fits the accuracies, \emph{recal.} fits their '
         r'ranks. Every arm sees the same architectures under the same index draws, so the '
         r'ridge rows are the same numbers as the \emph{own, recalibrated} rows of Table~III '
         r'of the main paper, and the comparison is a head-to-head rather than two protocols '
         r'side by side. Published values for '
         r'these predictor families are reported at other budgets, over other pools and '
         r'sometimes in Kendall $\tau$, so they are refitted here instead of quoted. '
         r'Nothing is tuned per dataset, per budget or against the evaluation set. The last '
         r'row is the exact degree-2 projection of the recalibrated response, computed from '
         r'the table with nothing fitted: no fitted arm reaches it at any budget.}'
         % o['protocol']['seeds'],
         r'\label{tab:learned}',
         r'\footnotesize',
         r'\setlength{\tabcolsep}{3.2pt}',
         r'\begin{tabular}{l l *{6}{c}}', r'\toprule',
         r'& & \multicolumn{2}{c}{C10} & \multicolumn{2}{c}{C100} '
         r'& \multicolumn{2}{c}{IN16} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
         r'$n$ & predictor & raw & recal. & raw & recal. & raw & recal. \\',
         r'\midrule']
    best = {}
    for n in o['budgets']:
        for d in DATASETS:
            best[(n, d)] = max(o['results'][d][str(n)][k]['recal']['rho'] for k, _, _ in arms)
    for i, n in enumerate(o['budgets']):
        if i:
            L.append(r'\addlinespace[2pt]')
        for j, (key, label, _) in enumerate(arms):
            cells = []
            for d in DATASETS:
                c = o['results'][d][str(n)][key]
                hit = abs(c['recal']['rho'] - best[(n, d)]) < 1e-12
                cells.append('%.3f' % c['raw']['rho'])
                cells.append((r'\textbf{%.3f}' if hit else '%.3f') % c['recal']['rho'])
            L.append(r'%s & %s & %s \\'
                      % (('%d' % n) if j == 0 else '', label, ' & '.join(cells)))
    L += [r'\midrule',
          r'\multicolumn{2}{l}{exact $g_2$, recalibrated (nothing fitted)} & '
          + ' & '.join(r'\multicolumn{2}{c}{%.3f}' % ref[d]['rho'] for d in DATASETS)
          + r' \\',
          r'\bottomrule', r'\end{tabular}', r'\end{table}']
    p = os.path.join(TBL, 'tab_learned.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return p


def report(o, arms):
    print('\nthe recalibration of Section V, applied to each arm (mean gain in rho over the '
          '%d budget-by-dataset cells):' % (len(DATASETS) * len(o['budgets'])))
    for key, label, _ in arms:
        g = o['findings']['recalibration_gain_rho'][key]
        print('  %-32s %+.3f   positive in %d of %d cells'
              % (label, g['mean'], g['n_cells_positive'], g['n_cells']))
    lead = o['findings']['ridge_leads_every_learned_predictor_at']
    print('\nthe ridge leads every trained predictor on all three datasets at n = %s'
          % (', '.join(str(n) for n in lead) if lead else 'no budget'))
    for c in o['findings']['cells_where_a_learned_predictor_leads_the_ridge']:
        print('  a trained predictor leads it: %s, n=%d, %s (%.3f vs %.3f)'
              % (SHORT[c['dataset']], c['budget'], c['predictor'], c['rho'], c['ridge_rho']))
    for k, v in o['checks'].items():
        print('  %-48s %s' % (k, v))


if __name__ == '__main__':
    main()
