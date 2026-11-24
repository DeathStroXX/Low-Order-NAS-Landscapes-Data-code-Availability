#!/usr/bin/env python3
"""s5_predictor_validation.py -- the LLM accuracy predictor, validated from the released record.

Supplementary Section~\\ref{sec:sm_fitness} cites this script for one claim: that the fidelity of
the pipeline's LoRA-tuned accuracy predictor can be rechecked from the *released* per-architecture
record alone, without the raw logs. `s11_proxy_fidelity.py` measures the same signal run by run
out of `logs/`, which is the right unit of analysis for a selection decision but needs the whole
log tree; this script repeats the measurement from

    data/gene_diversity_benchmark.csv

pooled over architectures rather than averaged over runs, so that a reader who has only the
released table can reproduce the number.

What is measured, per dataset (CIFAR-10 and CIFAR-100):

  1. rho        Spearman correlation between the predictor value the pipeline recorded
                (`pred`) and the measured 3-epoch accuracy (`acc3`), over every baseline and
                LLM-arm row of the record (the frozen control is analysed from its own logs
                elsewhere); every such row carries a predictor value. No accuracy filter, no
                de-duplication and no cache filter is applied, because the record already holds
                one row per architecture per run.
  2. CI         a percentile bootstrap interval for rho, resampling architectures (the pairs,
                not the two columns separately), 10,000 resamples at a fixed seed.
  3. tau        Kendall's tau-b on the same pairs, with its own interval from the *same*
                resamples, so the two statistics are reported on one bootstrap design.
  4. proxy      the free 1-epoch accuracy (`acc1`) scored on exactly those rows, and the
                paired difference rho(predictor) - rho(1-epoch) with an interval, which is the
                comparison Section~VIII of the main paper draws.
  5. high_band  what the predictor's own top band is worth: the mean measured accuracy of the
                architectures it scored at or above 90 per cent, against the population mean.

The predictor is not fitted here and nothing is tuned, so the only choice this script makes is
which architectures are scored; it is stated above and recorded in the output.

Conventions follow `s11_proxy_fidelity.py` (scipy for the rank statistics, results written as
JSON into `analysis/results/`) and `statsutil.py`. `statsutil` carries no bootstrap, so the
percentile bootstrap of `s15_prescreen.py` and `s30_recipe.py` is followed instead: resample with
replacement, take the 2.5th and 97.5th percentiles of the resampled statistic. It is drawn here
in fixed-size chunks from one seeded generator, so the interval is reproducible to the digit.

Writes:
  analysis/results/predictor_validation.json

    python3 analysis/s5_predictor_validation.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

#: `python3 -I` does not put the script's own directory on the path, so the sibling modules of
#: this package are made importable explicitly; the script then runs identically from the
#: project root and from inside `analysis/`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DATA, DATASETS, RES  # noqa: E402

#: the released per-architecture record -- not the raw logs
CSV = os.path.join(DATA, 'gene_diversity_benchmark.csv')
#: the predictor value the pipeline recorded, the measured reference, and the free proxy
PRED, REF, PROXY = 'pred', 'acc3', 'acc1'
#: bootstrap design: fixed resample count, fixed seed, fixed chunk size (the chunk size is part
#: of the seed state, so it is a constant rather than a tuning knob)
N_BOOT, SEED, CHUNK = 10000, 0, 500
#: the predictor's own top band, in per cent
HIGH_BAND = 90.0
DSN = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}


def spearman_rows(x, y):
    """Spearman correlation of each row of two equally shaped matrices.

    Resampling with replacement creates ties, so the midranks a resample implies have to be
    recomputed for that resample; this is `scipy.stats.spearmanr` restated row-wise so that
    10,000 of them cost one pass instead of 10,000 calls.
    """
    a = stats.rankdata(x, axis=1)
    b = stats.rankdata(y, axis=1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def bootstrap(pred, proxy, ref):
    """One bootstrap over architectures, shared by every statistic reported.

    The three columns are resampled together, by row index, so the predictor and the 1-epoch
    proxy are always scored on the same resampled set of architectures and their difference is
    paired. Returns the resampled rho for both signals and the resampled tau of the predictor.
    """
    n = len(ref)
    rng = np.random.default_rng(SEED)
    rho_p, rho_x, tau_p = [], [], []
    done = 0
    while done < N_BOOT:
        k = min(CHUNK, N_BOOT - done)
        idx = rng.integers(0, n, size=(k, n))
        rho_p.append(spearman_rows(pred[idx], ref[idx]))
        rho_x.append(spearman_rows(proxy[idx], ref[idx]))
        tau_p.append([stats.kendalltau(pred[i], ref[i]).statistic for i in idx])
        done += k
    return (np.concatenate(rho_p), np.concatenate(rho_x),
            np.concatenate([np.asarray(t, float) for t in tau_p]))


def pct_ci(v):
    """Percentile 95% interval of a bootstrap distribution, as in s15/s30."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    if not os.path.exists(CSV):
        print('released per-architecture record not found at %s' % CSV)
        return 1
    df = pd.read_csv(CSV)
    df = df[df['method'] != 'frozen'].rename(columns={'method': 'pipeline'})
    out = {'source': os.path.relpath(CSV, os.path.join(DATA, '..')),
           'n_rows': int(len(df)),
           'columns': {'predictor': PRED, 'reference': REF, 'proxy': PROXY},
           'selection': 'baseline and LLM-arm rows of the given dataset whose %s is not empty' % PRED,
           'bootstrap': {'resamples': N_BOOT, 'seed': SEED, 'chunk': CHUNK,
                         'method': 'percentile, resampling architectures'},
           'datasets': {}}

    for ds in DATASETS:
        sub = df[df['dataset'] == ds]
        rows = sub.dropna(subset=[PRED, REF, PROXY])
        pred = rows[PRED].to_numpy(float)
        proxy = rows[PROXY].to_numpy(float)
        ref = rows[REF].to_numpy(float)
        rho = float(stats.spearmanr(pred, ref).statistic)
        tau = float(stats.kendalltau(pred, ref).statistic)
        rho_x = float(stats.spearmanr(proxy, ref).statistic)
        b_rho, b_rhox, b_tau = bootstrap(pred, proxy, ref)
        band = pred >= HIGH_BAND
        out['datasets'][ds] = {
            'n_scored': int(len(rows)),
            'n_rows_dataset': int(len(sub)),
            'n_without_predictor': int(sub[PRED].isna().sum()),
            'pipelines_scored': sorted(rows['pipeline'].unique().tolist()),
            'pipelines_without_predictor':
                sorted(sub[sub[PRED].isna()]['pipeline'].unique().tolist()),
            'spearman': {'rho': rho, 'ci95': pct_ci(b_rho),
                         'se': float(np.nanstd(b_rho, ddof=1))},
            'kendall': {'tau': tau, 'ci95': pct_ci(b_tau),
                        'se': float(np.nanstd(b_tau, ddof=1))},
            'proxy_1epoch': {'rho': rho_x, 'ci95': pct_ci(b_rhox)},
            'predictor_minus_proxy': {'delta_rho': rho - rho_x,
                                      'ci95': pct_ci(b_rho - b_rhox)},
            'high_band': {'threshold': HIGH_BAND, 'n': int(band.sum()),
                          'mean_reference': float(ref[band].mean()) if band.any() else None,
                          'population_mean': float(ref.mean()),
                          'gap': float(ref[band].mean() - ref.mean()) if band.any() else None},
        }

    path = os.path.join(RES, 'predictor_validation.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    # ---------------- report ----------------
    print('The LLM accuracy predictor against the measured 3-epoch accuracy,')
    print('pooled over the architectures of the released record that carry a predictor value.')
    print('  record    %s (%d rows)' % (out['source'], out['n_rows']))
    print('  columns   predictor %s, reference %s, free proxy %s' % (PRED, REF, PROXY))
    print('  bootstrap %d resamples, percentile interval, seed %d' % (N_BOOT, SEED))
    print()
    print('  %-10s %6s  %-22s %-22s %s'
          % ('dataset', 'n', 'Spearman rho [95% CI]', "Kendall tau [95% CI]",
             '1-epoch rho'))
    for ds in DATASETS:
        e = out['datasets'][ds]
        s, k = e['spearman'], e['kendall']
        print('  %-10s %6d  %.3f [%.3f, %.3f]   %.3f [%.3f, %.3f]   %.3f'
              % (DSN[ds], e['n_scored'], s['rho'], s['ci95'][0], s['ci95'][1],
                 k['tau'], k['ci95'][0], k['ci95'][1], e['proxy_1epoch']['rho']))
    print()
    for ds in DATASETS:
        e = out['datasets'][ds]
        d = e['predictor_minus_proxy']
        print('  %-10s predictor minus 1-epoch proxy: %+.3f [%+.3f, %+.3f]'
              % (DSN[ds], d['delta_rho'], d['ci95'][0], d['ci95'][1]))
        print('  %-10s %d rows have no predictor value (%s)'
              % ('', e['n_without_predictor'], ', '.join(e['pipelines_without_predictor'])
                 or 'none'))
        h = e['high_band']
        if h['mean_reference'] is not None:
            print('  %-10s predictions >= %.0f%%: %d architectures, mean measured %.2f%% '
                  'against a population mean of %.2f%% (gap %+.2f)'
                  % ('', h['threshold'], h['n'], h['mean_reference'], h['population_mean'],
                     h['gap']))
    print('\nwrote analysis/results/predictor_validation.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
