"""How much of the exact degree-2 optimum each cheap ranker actually realizes.

Proposition 1 gives an upper bound in squared error, and the scale remark of Section IV-D says
that bound does
not transfer to a rank or a regret. What it leaves is a different and more useful reference
point: the squared-error-optimal degree-2 function g_2 is an *exhibited* element of G_2, so
whatever it attains in rank fidelity and in regret at a shortlist is an attainable value --
a lower bound on the best degree-2 ranker -- and every cheap ranker in this paper can be
placed against it exactly, with nothing fitted on the reference side.

This script does only that placement. It computes no landscape and runs no search: it reads
the exact ladder of s59_ceiling.py, the thirteen published zero-cost proxies as scored by
s43_zc_proxies.py, and the fitted models of s35_nb201.py and s54_spectrum_tables.py, and
expresses each as a percentage of what g_2 attains in the same metric on the same grid.

Why the denominator is g_2 on the recalibrated scale
----------------------------------------------------
rho is invariant to a monotone change of the response, so there is one rank-fidelity number
per degree, not one per scale -- but *which* degree-2 function the squared-error criterion
selects does depend on the scale, and on the recalibrated scale it selects a better ranker
(Section V). The best degree-2 ranker this paper can exhibit is therefore g_2 taken on the
recalibrated response, and that is the denominator. It is a lower bound on the degree-2
optimum rather than the optimum itself, so a realization above 100% is possible in principle
and would mean a ranker had found a better element of G_2 than the squared-error projection;
the script reports it rather than clipping it, and `checks` records whether it happened.

What `realization` is, and what it is not
-----------------------------------------
It is the ratio of a measured ranker's score to an attainable reference, in the metric a
budget is decided in. It is not a fraction of Proposition 1's ceiling, which lives in squared
error; the R^2 column of tab_bench(e) is that quantity and is kept separate here for exactly
the reason the scale remark gives. A ranker at 98% of the reference in rho may still sit
anywhere in regret, which is why both metrics are tabulated side by side.

Outputs
-------
results/realization.json      every number in the table
../tables/tab_realization.tex Table: each free ranker against the exact degree-2 optimum

    python3 analysis/s64_realization.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
TBL = os.path.join(ROOT, 'tables')

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}


def J(n):
    with open(os.path.join(RES, n + '.json')) as fh:
        return json.load(fh)


def ladder(cei, ds, scale, d):
    """The exact degree-d projection's row of s59_ceiling.py, on `raw` or `recalibrated`."""
    for row in cei['datasets'][ds][scale]['ladder']:
        if row['d'] == d:
            return row
    raise KeyError('no degree-%d row for %s/%s' % (d, ds, scale))


def main():
    cei, zc, spec, tgt = J('ceiling'), J('zc_nb201'), J('spectrum'), J('targets')
    scales = cei['datasets'][DATASETS[0]].keys()
    recal = 'recalibrated' if 'recalibrated' in scales else 'recal'

    # ------------------------------------------------------------------ the reference
    ref = {d: {'rho': ladder(cei, d, recal, 2)['rho'],
               'regret10': ladder(cei, d, recal, 2)['regret_at_10'],
               'rho_raw': ladder(cei, d, 'raw', 2)['rho'],
               'regret10_raw': ladder(cei, d, 'raw', 2)['regret_at_10'],
               'ceiling_r2': ladder(cei, d, 'raw', 2)['ceiling_r2_percent'] / 100.0}
           for d in DATASETS}

    # ------------------------------------------------------------------ the rankers
    rows = []

    def add(key, label, group, note, get):
        r = {'key': key, 'label': label, 'group': group, 'note': note, 'per_dataset': {}}
        for d in DATASETS:
            rho, reg = get(d)
            r['per_dataset'][d] = {
                'rho': rho, 'regret10': reg,
                'realization_rho': 100.0 * rho / ref[d]['rho'],
            }
        r['realization_rho_mean'] = sum(
            r['per_dataset'][d]['realization_rho'] for d in DATASETS) / len(DATASETS)
        r['rho_mean'] = sum(r['per_dataset'][d]['rho'] for d in DATASETS) / len(DATASETS)
        r['regret10_mean'] = sum(r['per_dataset'][d]['regret10'] for d in DATASETS) / len(DATASETS)
        rows.append(r)
        return r

    # the two exact projections, which are computed from the table and fit nothing
    add('g2_recal', r'exact $g_2$, recalibrated', 'exact',
        'the squared-error-optimal degree-2 function of the recalibrated response; the '
        'reference every other row is a percentage of',
        lambda d: (ref[d]['rho'], ref[d]['regret10']))
    add('g2_raw', r'exact $g_2$, raw accuracy', 'exact',
        'the same projection taken on raw accuracy: a different element of $\\mathcal{G}_2$, '
        'and a worse ranker',
        lambda d: (ref[d]['rho_raw'], ref[d]['regret10_raw']))
    add('g1_recal', r'exact $g_1$, recalibrated', 'exact',
        'main effects only: what the additive part of the landscape is worth',
        lambda d: (ladder(cei, d, recal, 1)['rho'], ladder(cei, d, recal, 1)['regret_at_10']))

    # label-free rankers: nothing from the target dataset is read
    lf = zc['label_free']
    add('proxy_best', 'best single proxy (oracle choice)', 'free',
        'the best of the thirteen on each dataset, chosen on the target -- not deployable, '
        'and an upper bound on what choosing one could give',
        lambda d: (lf[d]['proxy_rho'], lf[d]['proxy_regret10']))
    add('proxy_picked', 'the proxy an honest rule picks', 'free',
        'chosen without reading a target label, which is what a deployment can actually do',
        lambda d: (lf[d]['picked_rho'], lf[d]['picked_regret10']))
    add('transfer', 'transferred degree-2 surrogate', 'free',
        'fitted on the hardest other dataset; no target label',
        lambda d: (lf[d]['transfer_rho'], lf[d]['transfer_regret10']))
    add('fused', 'fusion: transferred surrogate + picked proxy', 'free',
        'the untuned 2:1 rank average of Section VII, which is the deployable object',
        lambda d: (lf[d]['fused_picked_rho'], lf[d]['fused_picked_regret10']))

    # trained predictors, on the same labels and the same draws (s68_learned_predictors.py)
    lp = None
    lp_path = os.path.join(RES, 'learned_predictors.json')
    if os.path.exists(lp_path):
        with open(lp_path) as fh:
            lp = json.load(fh)

    # fitted on the target's own labels, 1,000 of them
    rc = spec['recalibration']
    add('ridge_raw_1000', 'degree-2 ridge, raw, 1{,}000 labels', 'fitted',
        'the same model fitted on the target, without the recalibration',
        lambda d: (rc[d]['budgets']['1000']['own_raw']['rho'],
                   rc[d]['budgets']['1000']['own_raw']['regret10']))
    add('ridge_recal_1000', 'degree-2 ridge, recalibrated, 1{,}000 labels', 'fitted',
        'the procedure of Algorithm 1 up to the fusion step',
        lambda d: (rc[d]['budgets']['1000']['own_recal']['rho'],
                   rc[d]['budgets']['1000']['own_recal']['regret10']))

    #: the same comparison against what the performance-prediction literature fields, on the
    #: recalibrated response so that the only difference from the row above is the model class
    if lp is not None:
        LAB = {'rf': 'random forest', 'gbdt': 'gradient-boosted trees',
               'mlp_path': 'neural predictor, path encoding'}
        for key in ('rf', 'gbdt', 'mlp_path'):
            if key not in lp['results'][DATASETS[0]]['1000']:
                continue
            add('learned_' + key, LAB[key], 'learned',
                'refitted by s68_learned_predictors.py on the same 1,000 labels, under the '
                'same index draws as the ridge row above, and on the same recalibrated '
                'response; nothing tuned per dataset or against the evaluation set',
                lambda d, k=key: (lp['results'][d]['1000'][k]['recal']['rho'],
                                  lp['results'][d]['1000'][k]['recal']['regret10']))

    # ------------------------------------------------------------------ the R^2 side
    #: Proposition 1's quantity, kept apart from the rank column on purpose. The held-out R^2
    #: of the fitted degree-2 ridge against the exact ceiling it cannot exceed.
    rea = tgt['realization']
    r2 = {d: {'held_out_r2': rea[d]['held_out_r2'],
              'ceiling': rea[d]['ceiling_degree2'],
              'realized_percent': 100.0 * rea[d]['realized']} for d in DATASETS}

    out = {
        'n_arch': cei.get('n_arch'),
        'datasets': DATASETS,
        'reference': {'what': 'the exact degree-2 projection of the recalibrated response, '
                              'g_2; an attainable value, so a lower bound on the best '
                              'degree-2 ranker rather than the optimum itself',
                      'per_dataset': ref},
        'rows': rows,
        'squared_error': {'what': "Proposition 1's quantity: the held-out R^2 of the fitted "
                                  'degree-2 ridge as a share of the exact degree-2 ceiling, '
                                  'which is a different question from the rank column and is '
                                  'kept in its own block for that reason',
                          'per_dataset': r2},
    }
    exceed = [(r['key'], d) for r in rows for d in DATASETS
              if r['per_dataset'][d]['realization_rho'] > 100.0 + 1e-9 and r['group'] != 'exact']
    out['checks'] = {
        'reference_is_100_percent': all(
            abs(r['per_dataset'][d]['realization_rho'] - 100.0) < 1e-9
            for r in rows if r['key'] == 'g2_recal' for d in DATASETS),
        'any_ranker_beats_the_exhibited_optimum': bool(exceed),
        'rankers_above_the_reference': exceed,
        'recalibration_improves_the_projection': all(
            ref[d]['rho'] > ref[d]['rho_raw'] for d in DATASETS),
    }

    os.makedirs(RES, exist_ok=True)
    jp = os.path.join(RES, 'realization.json')
    with open(jp, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    tp = emit_table(out)
    report(out)
    print('\nwrote %s' % os.path.relpath(jp, ROOT))
    print('wrote %s' % os.path.relpath(tp, ROOT))
    return 0


# ------------------------------------------------------------------ the table
GROUP_HEAD = {
    'exact': r'\emph{exact, nothing fitted: what a degree-2 function attains}',
    'free': r'\emph{free rankers: no label from the target dataset}',
    'fitted': r"\emph{fitted on 1{,}000 of the target's own labels}",
    #: the trained predictor families the prediction literature actually fields, refitted by
    #: s68_learned_predictors.py on the same labels under the same index draws. Without this
    #: block the table compares a ridge against thirteen zero-cost proxies and leaves open
    #: whether anything trained would beat all of them by more.
    'learned': r'\emph{trained predictors, same 1{,}000 labels, same recalibration}',
}


def emit_table(o):
    L = []
    a = L.append
    a(r'\begin{table}[!t]')
    a(r'\centering')
    a(r'\caption{Every cheap ranker against the exact degree-2 optimum. $\rho$ is Spearman '
      r'rank fidelity over all 15{,}625 cells; \emph{real.} is that $\rho$ as a percentage of '
      r'the first row, averaged over the three datasets. That first row is the exact degree-2 '
      r'projection $g_2$ of the recalibrated response, an \emph{attainable} value computed '
      r'from the table with nothing fitted, so a lower bound on the best degree-2 ranker. It '
      r"is not the bound of the main paper's Proposition~1, which lives in squared error "
      r'and is the last block; its Remark~1 is why those are different questions. '
      r'Proxy scores are those of '
      r'NAS-Bench-Suite-Zero~\cite{krishnakumar2022nasbenchsuitezero}, unchanged. The trained '
      r'predictors see the same 1{,}000 labels under the same index draws as the ridge, so '
      r'the only difference from it is the model class; Table~\ref{tab:learned} gives them at '
      r'both budgets.}')
    a(r'\label{tab:realization}')
    a(r'\footnotesize')
    a(r'\setlength{\tabcolsep}{2.8pt}')
    a(r'\begin{tabular}{@{}l rrr r@{}}')
    a(r'\toprule')
    a(r'& \multicolumn{3}{c}{Spearman $\rho$} & \\')
    a(r'\cmidrule(lr){2-4}')
    a(r'Ranker & C10 & C100 & IN16 & real. \\')
    a(r'\midrule')
    last = None
    for r in o['rows']:
        if r['group'] != last:
            if last is not None:
                a(r'\addlinespace[2pt]')
            a(r'\multicolumn{5}{@{}l}{%s} \\' % GROUP_HEAD[r['group']])
            last = r['group']
        bold = r['key'] == 'g2_recal'
        cells = ' & '.join('%.3f' % r['per_dataset'][d]['rho'] for d in DATASETS)
        real = '%.0f\\%%' % r['realization_rho_mean']
        if bold:
            a(r'\quad \textbf{%s} & %s & \textbf{%s} \\'
              % (r['label'], ' & '.join(r'\textbf{%.3f}' % r['per_dataset'][d]['rho']
                                        for d in DATASETS), real))
        else:
            a(r'\quad %s & %s & %s \\' % (r['label'], cells, real))
    a(r'\addlinespace[2pt]\midrule')
    a(r'\multicolumn{5}{@{}l}{\emph{the same rows in the decision metric: regret at ten, '
      r'accuracy points}} \\')
    #: the scale remark is why this block is here at all, and why it is not ordered as the rho
    #: block is. The full set of rows is in results/realization.json.
    for key in ('g2_recal', 'g2_raw', 'transfer', 'fused', 'ridge_recal_1000',
                'learned_gbdt'):
        if not any(x['key'] == key for x in o['rows']):
            continue
        r = next(x for x in o['rows'] if x['key'] == key)
        a(r'\quad %s & %s & \\'
          % (r['label'], ' & '.join('%.2f' % r['per_dataset'][d]['regret10']
                                    for d in DATASETS)))
    se = o['squared_error']['per_dataset']
    if all(se[d].get('held_out_r2') is not None for d in DATASETS):
        a(r'\addlinespace[2pt]\midrule')
        a(r'\multicolumn{5}{@{}l}{\emph{the other question: squared error, against '
          r"the main paper's Proposition~1}} \\")
        a(r'\quad degree-2 ceiling $A_{\le 2}$ & %s & \\'
          % ' & '.join('%.3f' % se[d]['ceiling'] for d in DATASETS))
        a(r'\quad held-out $R^{2}$ of the ridge & %s & \\'
          % ' & '.join('%.2f' % se[d]['held_out_r2'] for d in DATASETS))
        a(r'\quad realized & %s & \\'
          % ' & '.join('%.0f\\%%' % se[d]['realized_percent'] for d in DATASETS))
    a(r'\bottomrule')
    a(r'\end{tabular}')
    a(r'\end{table}')
    p = os.path.join(TBL, 'tab_realization.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return p


def report(o):
    print('every cheap ranker as a percentage of the exact degree-2 optimum it could reach')
    print('(rho, over all %s cells; the reference is the exact g_2 of the recalibrated '
          'response)\n' % '{:,}'.format(o['n_arch']) if o['n_arch'] else '')
    print('  %-44s %7s %7s %7s %7s' % ('ranker', *[SHORT[d] for d in DATASETS], 'real.'))
    for r in o['rows']:
        print('  %-44s %7.3f %7.3f %7.3f %6.0f%%'
              % (r['label'].replace('{,}', ',').replace('$', '').replace('\\', ''),
                 *[r['per_dataset'][d]['rho'] for d in DATASETS],
                 r['realization_rho_mean']))
    c = o['checks']
    print('\nchecks: the reference row is 100%% by construction: %s | recalibration selects a '
          'better' % c['reference_is_100_percent'])
    print('        degree-2 ranker on every dataset: %s | a measured ranker passes the '
          'exhibited' % c['recalibration_improves_the_projection'])
    print('        degree-2 optimum: %s%s' % (
        c['any_ranker_beats_the_exhibited_optimum'],
        '' if not c['rankers_above_the_reference']
        else ' (%s)' % ', '.join('%s/%s' % (k, SHORT[d])
                                 for k, d in c['rankers_above_the_reference'])))


if __name__ == '__main__':
    sys.exit(main())
