"""Table II: the second complete factorial, the measured nulls and the certificates.

  tables/tab_bench.tex     everything that is exact and not NAS-Bench-201's accuracy: a second
                           complete factorial, a measured noise floor, the implementation
                           certificates, and how much of the ceiling a free model reaches

It is assembled from four result files rather than recomputed here:

  results/benchmarks.json  NAS-Bench-Macro, the 3^8 factorial and its measured null (s57)
  results/hardware.json    the nine HW-NAS-Bench columns on the 5^6 grid (s58)
  results/targets.json     the parameter and FLOP certificates, and the realized ceiling (s56)
  results/spectrum.json    NAS-Bench-201's accuracy spectrum, as the reference row (s53)

    python3 analysis/s60_tables.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TBL = os.path.join(HERE, '..', 'tables')

DS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}


def J(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def tab_bench(B, H, T, S):
    acc = B['accuracy']
    nul = B['null']
    mn = B['measured_null']
    cum = acc['cumulative']
    ncum = nul['cumulative']
    mcum = mn['cumulative_mean']
    hs = H['summary']
    cert5 = T['certificate']
    cert3 = B['certificates']
    rea = T['realization']

    L = [r'\begin{table}[!htb]', r'\centering',
         r'\caption{Everything else the exactness condition reaches. \textbf{(a)} A second '
         r'complete factorial from another group: NAS-Bench-Macro\cite{su2021mcts}, all '
         r'$3^{8}$ architectures trained from scratch three times. \textbf{(b)} Two of those '
         r'trainings differ only by noise, so their difference is a function on the same grid '
         r'whose entire content is noise, which measures the null instead of deriving it. '
         r'\textbf{(c)} The same measurement on the $5^{6}$ grid, from the hardware columns '
         r'of\cite{li2021hwnasbench}, additive over the six edges up to measurement error. '
         r'\textbf{(d)} Parameter and FLOP counts are additive by construction, so an exact '
         r'decomposition must return pure order one.}',
         r'\label{tab:bench}', r'\footnotesize',
         r'\setlength{\tabcolsep}{3.0pt}',
         r'\begin{tabular}{l rrrrr}', r'\toprule',
         r'\multicolumn{6}{l}{\emph{(a) NAS-Bench-Macro, $3^{8}=6{,}561$ cells, CIFAR-10}} \\',
         r'order & 1 & 2 & 3 & $\le 2$ & $\le 3$ \\',
         r'\midrule',
         r'accuracy & %.1f & %.1f & %.1f & \textbf{%.1f} & \textbf{%.1f} \\'
         % (acc['per_order'][0], acc['per_order'][1], acc['per_order'][2], cum[1], cum[2]),
         r'\addlinespace[1pt]',
         r'\multicolumn{6}{l}{\emph{(b) the null, measured rather than derived}} \\',
         r'seed difference & %.1f & %.1f & %.1f & %.1f & %.1f \\'
         % (mn['per_order_mean'][0], mn['per_order_mean'][1], mn['per_order_mean'][2],
            mcum[1], mcum[2]),
         r'white-noise null & %.1f & %.1f & %.1f & %.1f & %.1f \\'
         % (nul['per_order'][0], nul['per_order'][1], nul['per_order'][2], ncum[1], ncum[2]),
         r'\multicolumn{6}{l}{\quad\footnotesize the two agree to %.1f points at every one of '
         r'the eight orders} \\' % mn['max_abs_deviation_from_analytic_null'],
         r'\multicolumn{6}{l}{\footnotesize skewness $%.2f$, top-decile spread %.1f\%%, '
         r'$\le 2$ recalibrated %.1f\%%} \\'
         % (acc['compression']['skewness'], 100 * acc['compression']['top_decile_spread'],
            B['recalibrated']['cumulative'][1]),
         r'\addlinespace[2pt]\midrule',
         r'\multicolumn{6}{l}{\emph{(c) a measured noise floor on the $5^{6}$ grid: '
         r'HW-NAS-Bench}} \\',
         r'& \multicolumn{2}{c}{order 1} & \multicolumn{2}{c}{order 6} & $n$ \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'measured on the device & %.1f & %.1f & %.2f & %.2f & %d \\'
         % (hs['device_order1_min'], hs['device_order1_max'],
            hs['device_order6_min'], hs['device_order6_max'], hs['n_device_columns']),
         r'from a cost model & %.1f & %.1f & %.3f & %.3f & %d \\'
         % (hs['modelled_order1_min'], hs['modelled_order1_max'], 0.0,
            hs['modelled_order6_max'], hs['n_modelled_columns']),
         r'accuracy, same grid & %.1f & %.1f & %.2f & %.2f & %d \\'
         % (100 * min(S['spectrum'][d]['share']['1'] for d in DS),
            100 * max(S['spectrum'][d]['share']['1'] for d in DS),
            hs['accuracy_order6_min'], hs['accuracy_order6_max'], 3),
         r'\addlinespace[2pt]\midrule',
         r'\multicolumn{6}{l}{\emph{(d) exactness certificates: an additive target must '
         r'return pure order one}} \\',
         r'& \multicolumn{2}{c}{order one (\%)} & \multicolumn{2}{c}{above it (\%)} & \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'params, $5^{6}$ / $3^{8}$ & %.6f & %.6f & %.0e & %.0e & \\'
         % (100 * cert5['params']['order1'], cert3['params']['order1'],
            100 * cert5['params']['orders_2_to_6'], cert3['params']['above_order1']),
         r'FLOPs, $5^{6}$ / $3^{8}$ & %.6f & %.6f & %.0e & %.0e & \\'
         % (100 * cert5['flop']['order1'], cert3['flops']['order1'],
            100 * cert5['flop']['orders_2_to_6'], cert3['flops']['above_order1']),
         r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(L)


def tab_bench_main(full):
    """Blocks (a)-(c) of tab_bench for the main paper: the second benchmark, the measured null
    and the hardware noise floor. Cut from the same generated text, so the two cannot differ."""
    a = full.index(r'\begin{tabular}')
    b = full.index(r'\multicolumn{6}{l}{\emph{(d)')
    body = full[a:b].rstrip()
    if body.endswith(r'\addlinespace[2pt]\midrule'):
        body = body[:-len(r'\addlinespace[2pt]\midrule')].rstrip()
    return '\n'.join([
        r'\begin{table}[!t]', r'\centering',
        r'\caption{Low-order concentration on NAS-Bench-Macro, a measured null and a '
        r'hardware noise floor}',
        r'\label{tab:bench_main}', r'\footnotesize', r'\setlength{\tabcolsep}{3.2pt}',
        body.replace(r'\begin{tabular}{', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}', 1),
        r'\bottomrule',
        r'\multicolumn{6}{@{}p{\linewidth}@{}}{\scriptsize Shares of variance (\%). (a) '
        r'NAS-Bench-Macro, CIFAR-10. (b) Difference between two trainings '
        r'of the same architectures, beside the analytic white-noise null. (c) HW-NAS-Bench '
        r'device measurements on the NAS-Bench-201 grid; ranges over the measured columns. '
        r'Supplementary Table~S15 adds the exactness certificates.} \\',
        r'\end{tabular*}', r'\end{table}', ''])


def main():
    B, H, T, S = (J('benchmarks.json'), J('hardware.json'), J('targets.json'),
                  J('spectrum.json'))
    full = tab_bench(B, H, T, S)
    open(os.path.join(TBL, 'tab_bench.tex'), 'w').write(full)
    open(os.path.join(TBL, 'tab_bench_main.tex'), 'w').write(tab_bench_main(full))
    print('wrote tables/tab_bench.tex, tables/tab_bench_main.tex')
    return 0


if __name__ == '__main__':
    sys.exit(main())
