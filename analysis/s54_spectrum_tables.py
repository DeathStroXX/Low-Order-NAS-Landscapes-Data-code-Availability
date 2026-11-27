"""Table I, Table IV and Fig. 1 from the spectrum: the landscape result and what it costs.

Reads `analysis/results/spectrum.json` (written by `s53_spectrum.py`) and renders

  tables/tab_spectrum.tex   the exact amplitude spectrum against the white-noise null, and
                            the same spectrum after monotone recalibration
  tables/tab_recal.tex      recalibrating the target's own labels against transferring a prior
                            from a harder dataset, at four budgets, on both metrics, with the
                            uncompressed space as the control
  figures/fig_spectrum.pdf  the spectrum as a figure: observed against the null, raw against
                            recalibrated

The figure's three dataset hues are fixed in this order and were checked for colour-vision
separation before use (worst adjacent pair dE 27.1 protan, 15.3 tritan, against a light
surface). The white-noise null is a reference level rather than a fourth series, so it is drawn
as a recessive dashed step and never competes with the data for a hue.

    python3 analysis/s54_spectrum_tables.py
"""
import json, math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TBL = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')

DS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}
#: validated categorical order; see the module docstring
HUE = {'cifar10': '#4C6EDB', 'cifar100': '#D4722B', 'ImageNet16-120': '#9B51C9'}
INK, MUTED, GRID = '#1a1a1a', '#5b5b5b', '#d8d8d4'


#: Main-paper tables follow the IEEE screening rule: a short title (no leading article, no final
#: period) and every protocol, provenance and column gloss in a note row under \bottomrule. The
#: table is set at full column width (tabular*) so the note row can span it exactly.
def note_row(ncols, text):
    return r'\multicolumn{%d}{@{}p{\linewidth}@{}}{\scriptsize %s} \\' % (ncols, text)


def J(name='spectrum.json'):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def tab_spectrum(S, C=None):
    sp, rc, nl = S['spectrum'], S['spectrum_recalibrated'], S['null']
    L = [r'\begin{table}[!htb]', r'\centering',
         r'\caption{Exact amplitude spectrum of NAS-Bench-201, by interaction order}',
         r'\label{tab:spectrum}', r'\footnotesize',
         r'\setlength{\tabcolsep}{3.4pt}',
         r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l rrr c rrr}', r'\toprule',
         r'& \multicolumn{3}{c}{share of variance, raw} & Null & '
         r'\multicolumn{3}{c}{after recalibration} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){6-8}',
         r'Order & C10 & C100 & IN16 & & C10 & C100 & IN16 \\', r'\midrule']
    for k in range(1, 7):
        L.append(r'%d & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\'
                 % (k, *[100 * sp[d]['share'][str(k)] for d in DS], 100 * nl[str(k)],
                    *[100 * rc[d]['share'][str(k)] for d in DS]))
    L.append(r'\addlinespace[2pt]\midrule')
    L.append(r'\multicolumn{8}{l}{\emph{cumulative: the $R^{2}$ ceiling of any degree-$d$ '
             r'predictor}} \\')
    for k in (1, 2, 3):
        L.append(r'$\le %d$ & \textbf{%.1f} & \textbf{%.1f} & \textbf{%.1f} & %.1f & '
                 r'%.1f & %.1f & %.1f \\'
                 % (k, *[100 * sp[d]['cumulative'][str(k)] for d in DS],
                    100 * sum(nl[str(j)] for j in range(1, k + 1)),
                    *[100 * rc[d]['cumulative'][str(k)] for d in DS]))
    L.append(r'\addlinespace[2pt]\midrule')
    L.append(r'\multicolumn{8}{l}{\emph{how compressed the accuracy scale is}} \\')
    L.append(r'skewness & %.2f & %.2f & %.2f & --- & \multicolumn{3}{c}{---} \\'
             % tuple(S['compression'][d]['skew'] for d in DS))
    L.append(r'top-decile spread (\%%) & %.1f & %.1f & %.1f & --- & \multicolumn{3}{c}{---} \\'
             % tuple(100 * S['compression'][d]['top_decile_spread'] for d in DS))
    if C is not None:
        lad = {d: {'raw': {r['d']: r for r in C['datasets'][d]['raw']['ladder']},
                   'rec': {r['d']: r for r in C['datasets'][d]['recalibrated']['ladder']}}
               for d in DS}
        L.append(r'\addlinespace[2pt]\midrule')
        L.append(r'\multicolumn{8}{l}{\emph{what $g_d$ attains: the same bound in the '
                 r'decision metrics}} \\')
        for k in (1, 2, 3):
            L.append(r'$\rho(g_{%d})$ & %.3f & %.3f & %.3f & --- & %.3f & %.3f & %.3f \\'
                     % (k, *[lad[d]['raw'][k]['rho'] for d in DS],
                        *[lad[d]['rec'][k]['rho'] for d in DS]))
        for k in (1, 2, 3):
            L.append(r'regret$(g_{%d})$ & %.2f & %.2f & %.2f & --- & %.2f & %.2f & %.2f \\'
                     % (k, *[lad[d]['raw'][k]['regret_at_10'] for d in DS],
                        *[lad[d]['rec'][k]['regret_at_10'] for d in DS]))
    L += [r'\bottomrule',
          note_row(8, r'Shares of accuracy variance (\%). Null: white noise, from the '
                      r'$\binom{6}{k}4^{k}$ basis directions at order $k$. Recal.: after a '
                      r'monotone recalibration. $g_d$: the squared-error-optimal degree-$d$ '
                      r'function; regret at a shortlist of ten, in accuracy points.'),
          r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(L)


def recal_margin(R):
    """Smallest win of own_recal over transfer_raw, in standard errors of the difference
    (the arms share their draws, so treating them as independent is conservative)."""
    z = []
    for v in R.values():
        for b in v['budgets'].values():
            o, t = b['own_recal'], b['transfer_raw']
            if 'rho_se' not in o:
                return None
            z.append((o['rho'] - t['rho']) / (o['rho_se'] ** 2 + t['rho_se'] ** 2) ** 0.5)
            z.append((t['regret10'] - o['regret10']) /
                     (o['regret10_se'] ** 2 + t['regret10_se'] ** 2) ** 0.5)
    return min(z)


def tab_recal(S):
    R = S['recalibration']
    zmin = recal_margin(R)
    L = [r'\begin{table}[!htb]', r'\centering',
         r'\caption{Own recalibrated labels against a transferred raw-scale prior, with an '
         r'uncompressed control}',
         r'\label{tab:recal}', r'\footnotesize',
         r'\setlength{\tabcolsep}{3.2pt}',
         r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l r rrr c rrr}', r'\toprule',
         r'& & \multicolumn{3}{c}{Spearman $\rho$} & & \multicolumn{3}{c}{regret at ten} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){7-9}',
         r'Target & $n$ & raw & \textbf{recal.} & transf. & & raw & \textbf{recal.} & transf. '
         r'\\', r'\midrule']
    for d in DS:
        L.append(r'\multicolumn{9}{l}{\emph{%s; transfer from %s}} \\'
                 % (SHORT[d], SHORT[R[d]['transfer_source']]))
        for n in ('100', '250', '500', '1000'):
            b = R[d]['budgets'][n]
            L.append(r'\quad & %s & %.3f & \textbf{%.3f} & %.3f & & %.2f & \textbf{%.2f} & '
                     r'%.2f \\'
                     % (n, b['own_raw']['rho'], b['own_recal']['rho'],
                        b['transfer_raw']['rho'], b['own_raw']['regret10'],
                        b['own_recal']['regret10'], b['transfer_raw']['regret10']))
    if 'fractalnet' in S:
        L.append(r'\addlinespace[2pt]\midrule')
        L.append(r'\multicolumn{9}{l}{\emph{control: FractalNet, accuracies not compressed}} '
                 r'\\')
        for ds, v in S['fractalnet'].items():
            for n, b in sorted(v['budgets'].items(), key=lambda kv: int(kv[0])):
                L.append(r'\quad %s & %s & %.3f & %.3f & --- & & %.2f & %.2f & --- \\'
                         % (SHORT.get(ds, ds), n, b['own_raw']['rho'], b['own_recal']['rho'],
                            b['own_raw']['regret10'], b['own_recal']['regret10']))
    L += [r'\bottomrule',
          note_row(9, r'Same degree-2 ridge and fit sets, scored on held-out architectures, 20 '
                      r'draws. raw, recal.: the target\textquotesingle s own labels as measured '
                      r'and recalibrated; transf.: a raw-scale prior fitted on the source named. '
                      + (r'Every recalibration win is at least %.1f standard errors of the '
                         r'difference; ' % (math.floor(10 * zmin) / 10) if zmin else '')
                      + r'Supplementary Table~S17 gives the standard errors and two further '
                        r'transfer arms. Control: FractalNet, accuracies not compressed.'),
          r'\end{tabular*}', r'\end{table}', '']
    return '\n'.join(L)


def figure(S):
    sp, rc, nl = S['spectrum'], S['spectrum_recalibrated'], S['null']
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
    orders = list(range(1, 7))
    w = 0.26
    for ax, src, title in ((axes[0], sp, 'as measured'),
                           (axes[1], rc, 'after recalibration')):
        for i, d in enumerate(DS):
            ax.bar([o + (i - 1) * w for o in orders],
                   [100 * src[d]['share'][str(k)] for k in orders],
                   width=w - 0.045, color=HUE[d], label=SHORT[d], zorder=3,
                   linewidth=0)
        ax.step([o - 1.5 * w for o in orders] + [6 + 1.5 * w],
                [100 * nl[str(k)] for k in orders] + [100 * nl['6']],
                where='post', color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)),
                zorder=4, label='white-noise null')
        ax.set_title(title, fontsize=6.6, color=INK, pad=3)
        ax.set_xlabel('interaction order', fontsize=6.4, color=MUTED)
        ax.set_xticks(orders)
        ax.tick_params(labelsize=6.0, colors=MUTED, length=2, width=0.6)
        ax.grid(axis='y', color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
            ax.spines[s].set_linewidth(0.6)
    axes[0].set_ylabel('% of accuracy variance', fontsize=6.4, color=MUTED)
    axes[0].legend(fontsize=5.6, frameon=False, loc='upper right', labelcolor=INK,
                   handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig_spectrum.' + ext), dpi=400)
    plt.close(fig)


def tab_recal_full(S):
    """All five arms of the recalibrate-or-transfer comparison, mean +- standard error."""
    R = S['recalibration']
    arms = [('own_recal', 'own, recal.'), ('transfer_raw', 'transf., raw'),
            ('transfer_recal', 'transf., recal.'), ('combined', 'both')]
    L = [r'\begin{table}[h]', r'\centering\footnotesize',
         r'\caption{Recalibrate, transfer, or both, on NAS-Bench-201: the same degree-2 ridge on '
         r'the same fit sets, fitted on the target\textquotesingle s own recalibrated labels, on '
         r'the transfer source\textquotesingle s labels raw or recalibrated, or on both '
         r'(rank-average of the two recalibrated predictions); mean $\pm$ standard error over 20 '
         r'draws. Produced by \texttt{analysis/s53\_spectrum.py}.}',
         r'\label{tab:recal_full}', r'\setlength{\tabcolsep}{2.6pt}',
         r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{l r cccc cccc}', r'\toprule',
         r'& & \multicolumn{4}{c}{Spearman $\rho$} & \multicolumn{4}{c}{regret at ten} \\',
         r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
         r'Target & $n$ & ' + ' & '.join(a for _, a in arms) + ' & '
         + ' & '.join(a for _, a in arms) + r' \\', r'\midrule']
    for d in DS:
        L.append(r'\multicolumn{10}{l}{\emph{%s; transfer from %s}} \\'
                 % (SHORT[d], SHORT[R[d]['transfer_source']]))
        for n in ('100', '250', '500', '1000'):
            b = R[d]['budgets'][n]
            rho = ' & '.join(r'%.3f\tiny{$\pm$%.3f}' % (b[k]['rho'], b[k].get('rho_se', 0)) for k, _ in arms)
            reg = ' & '.join(r'%.2f\tiny{$\pm$%.2f}' % (b[k]['regret10'], b[k].get('regret10_se', 0)) for k, _ in arms)
            L.append(r'\quad & %s & %s & %s \\' % (n, rho, reg))
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    return '\n'.join(L)


def main():
    S = J()
    try:
        C = J('ceiling.json')
    except FileNotFoundError:
        C = None
        print('note: results/ceiling.json is absent, so the decision-metric block of '
              'Table I is omitted. Run analysis/s59_ceiling.py first.')
    open(os.path.join(TBL, 'tab_spectrum.tex'), 'w').write(tab_spectrum(S, C))
    open(os.path.join(TBL, 'tab_recal.tex'), 'w').write(tab_recal(S))
    if 'transfer_recal' in next(iter(S['recalibration'].values()))['budgets']['100']:
        open(os.path.join(TBL, 'tab_recal_full.tex'), 'w').write(tab_recal_full(S))
    figure(S)
    print('wrote tables/tab_spectrum.tex, tables/tab_recal.tex, figures/fig_spectrum.pdf')
    return 0


if __name__ == '__main__':
    sys.exit(main())
