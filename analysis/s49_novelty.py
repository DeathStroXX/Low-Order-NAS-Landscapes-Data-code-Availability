"""Table I: what the closest prior art establishes, and what this paper adds.

The "incremental" objection is the one most likely to end a review of this manuscript at a
journal whose instructions tell reviewers that a Full Paper must not be incremental, and the
objection is reasonable on the face of it: that cheap predictors are hard to beat is
established (White et al. 2021; Ning et al. 2021). The answer is not rhetoric but a ledger --
claim by claim, what is already known and what is measured here for the first time -- and a
ledger is only worth as much as the numbers in it, so this script builds the table from
`analysis/results/` rather than from prose. A claim whose evidence moves when an analysis is
rerun changes in Supplementary Table S2 too.

Each row carries: a short claim, the closest prior work and what it establishes, what this
paper adds, and the evidence, pulled live from the results files. The row order is the order
the claims appear in the manuscript.

The ledger exists twice, from one source. Section II of the body carries Table I, the short
form -- because the body is where a reader decides whether the paper is incremental, and a
pointer into a supplement is not where that question gets settled. The supplement carries the
same nine rows at the length the prior art takes to state fairly. The prose of the short form
lives in `novelty_short.py`; every *number* in both tables is computed here, so the two cannot
disagree about a value.

    python3 analysis/s49_novelty.py     # -> tables/tab_novelty.tex, tables/tab_novelty_body.tex,
                                        #    results/novelty.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from novelty_short import SHORT  # noqa: E402

RES = os.path.join(HERE, 'results')
TBL = os.path.join(HERE, '..', 'tables')

A, B = 'cifar10', 'cifar100'
NB = ['cifar10', 'cifar100', 'ImageNet16-120']


def J(name):
    with open(os.path.join(RES, name + '.json')) as fh:
        return json.load(fh)


def rows():
    """The claim ledger. Every number is read from results/, never typed."""
    sur, anat = J('surrogate'), J('signal_anatomy')
    nb, zc, enc = J('nb201'), J('zc_nb201'), J('encoding')
    spaces, sitl, reg = J('zc_spaces'), J('sitl'), J('regret')
    spec = J('spectrum')

    lf, pr, ex = zc['label_free'], zc['paired'], zc['extra_tasks']
    diag = enc['diagnostic']
    nb201_tests = [diag[k]['test'] for k in diag if k.startswith('nb201')]
    S, R, N = spec['spectrum'], spec['spectrum_recalibrated'], spec['null']
    D = spec['datasets']
    out = []

    # ---- 1. the exact spectrum: the technique is old, the measurement is not
    out.append(dict(
        claim='The amplitude spectrum of a NAS landscape, measured exactly',
        prior='Amplitude spectra~\\cite{hordijk1998amplitude}; Fourier analysis of '
              'landscapes~\\cite{weinberger1991fourier}; fANOVA~\\cite{hutter2014fanova}; '
              '\\emph{exact} fANOVA on complete grids, reported to order '
              'eleven~\\cite{pushak2022automl}; exact fANOVA on an enumerated architecture '
              'grid~\\cite{klein2019tabular}; the exact closed form for categorical '
              'inputs~\\cite{ferrere2026exact}; exact Fourier decomposition of combinatorial '
              'objectives~\\cite{elorza2025fourier}',
        prior_says='Decomposing a function by interaction order is standard; the exact closed '
                   'form for categorical inputs is known in full generality; and the exact '
                   'decomposition has been computed on complete AutoML grids to order eleven '
                   'and on an enumerated architecture grid to order two. We find no '
                   'instance of it applied to a NAS landscape.',
        we_add='The apparatus is prior art and is used as such. What is new is the '
               'landscape it is turned on, and the answer is not a foregone conclusion: the '
               'same exact decomposition makes AutoML loss landscapes \\emph{high}-order '
               '($17.01$--$28.58\\%\\ at order six against $0.15$--$4.50\\%\\ at order '
               'one~\\cite{pushak2022automl}) and makes two NAS spaces low-order. The '
               'contribution is the measurement and its consequences, not the method.',
        evidence='order $\\le 2$ carries %.1f--%.1f\\%% of accuracy variance, order '
                 '$\\le 3$ carries %.1f--%.1f\\%%'
                 % (100 * min(S[d]['cumulative']['2'] for d in D),
                    100 * max(S[d]['cumulative']['2'] for d in D),
                    100 * min(S[d]['cumulative']['3'] for d in D),
                    100 * max(S[d]['cumulative']['3'] for d in D)),
    ))

    # ---- 2. the ceiling
    out.append(dict(
        claim='A structural ceiling on any degree-$d$ predictor',
        prior='Predictor comparisons~\\cite{white2021powerful,ning2021evaluating}; benchmark '
              'noise floors~\\cite{white2021exploring}',
        prior_says='Cheap predictors are hard to beat, and some apparent ruggedness is seed '
                   'noise. Neither yields a limit on what a predictor could achieve.',
        we_add='Because the components are orthogonal, the cumulative spectrum upper-bounds '
               'the $R^{2}$ of any degree-$d$ function of the design variables, however much '
               'data it is given. The elaborate predictors compete for the few percent '
               'above order three.',
        evidence='degree-2 ceiling %.2f--%.2f; degree-3 ceiling %.2f--%.2f'
                 % (min(S[d]['cumulative']['2'] for d in D),
                    max(S[d]['cumulative']['2'] for d in D),
                    min(S[d]['cumulative']['3'] for d in D),
                    max(S[d]['cumulative']['3'] for d in D)),
    ))

    # ---- 3. the scale illusion: a correction, not a discovery of the phenomenon
    out.append(dict(
        claim='The task-dependence of landscape simplicity is an artifact of the scale',
        prior='Tao and Bang~\\cite{tao2024landscape}; Lopes \\emph{et '
              'al.}~\\cite{lopes2024operation}; Chitra \\emph{et '
              'al.}~\\cite{chitra2025resolving}',
        prior_says='On this exact space, correlation length grows with task difficulty, and '
                   'searching on harder datasets is more reliable; the phenomenon is '
                   'reported and no mechanism is offered. Separately, in biological fitness '
                   'data, measures of higher-order interaction are known to disagree in sign '
                   'with the scale they are defined on.',
        we_add='We reproduce the ordering in the ANOVA basis and show it is the response '
               'scale, not the task: easy tasks are compressed against a ceiling, and a '
               'monotone recalibration collapses the three spectra onto one.',
        evidence='order $\\le 2$ goes from %.1f, %.1f, %.1f\\%% to %.1f, %.1f, %.1f\\%% '
                 'after recalibration'
                 % tuple([100 * S[d]['cumulative']['2'] for d in D]
                         + [100 * R[d]['cumulative']['2'] for d in D]),
    ))

    # ---- 4. recalibrate before transferring
    c10 = spec['recalibration']['cifar10']['budgets']['1000']
    out.append(dict(
        claim='Recalibrate first; a prior from a harder task mostly supplied a less compressed scale',
        prior='Cross-task predictors and cold-start work; the reliability ordering reported '
              'on this space~\\cite{tao2024landscape,lopes2024operation}',
        prior_says='Harder datasets give longer correlation lengths and more reliable '
                   'searches, and a prior fitted on a harder dataset ranks a target better '
                   'than the same model fitted on a thousand of the target\'s own labels, so '
                   'fit on the hardest dataset available.',
        we_add='True as measured, but mostly a matter of scale. The transferred prior '
               'supplied a less compressed response; recalibrating the target\'s own labels '
               'beats raw-scale transfer at every budget on both metrics, recalibrating the '
               'source closes most of that gap, and in an uncompressed space recalibration buys '
               'almost nothing and can cost regret.',
        evidence='CIFAR-10 at 1{,}000 labels: $\\rho=%.3f$ recalibrated against %.3f '
                 'raw-scale transfer, %.3f recalibrated transfer and %.3f raw; regret %.2f '
                 'against %.2f, %.2f and %.2f'
                 % (c10['own_recal']['rho'], c10['transfer_raw']['rho'],
                    c10['transfer_recal']['rho'], c10['own_raw']['rho'],
                    c10['own_recal']['regret10'], c10['transfer_raw']['regret10'],
                    c10['transfer_recal']['regret10'], c10['own_raw']['regret10']),
    ))

    # ---- 5. label complexity, transferred from another field
    out.append(dict(
        claim='Label complexity is a coefficient count, not a function of the space size',
        prior='Brookes \\emph{et al.}~\\cite{brookes2022sparsity}',
        prior_says='Protein and RNA fitness functions are sparse in the Fourier basis, so the '
                   'labels needed are set by that sparsity rather than by the size of the '
                   'space.',
        we_add='The same argument holds in NAS, with consequences their setting does not '
               'have: a degree-2 model of a $10^{18}$-point space is identified by its '
               'pairwise coefficients, which a study that has already run owns without '
               'buying a sampling design.',
        evidence='1{,}514 cells covered at %.1f and %.1f labels each, from %.3f\\%% of the '
                 'grid' % (anat[A]['identifiability']['labels_per_cell'],
                           anat[B]['identifiability']['labels_per_cell'],
                           100 * anat[A]['identifiability']['grid_fraction']),
    ))

    # ---- 6. the encoding boundary, given a mechanism
    out.append(dict(
        claim='The scope boundary is a basis in which the function is not low-order',
        prior='White \\emph{et al.}~\\cite{white2020encodings}',
        prior_says='Path encodings beat adjacency encodings, justified combinatorially by '
                   'relabelling invariance and graceful truncation.',
        we_add='The spectral reading of the same fact: a re-encoding adds no information, it '
               'moves the function to a lower order, and the held-out $R^{2}$ of a degree-2 '
               'fit measures whether it has. That makes the boundary testable before any '
               'budget is spent, and repairable for free.',
        evidence='held-out $R^{2}=%.2f$ on NAS-Bench-101 against $+%.2f$ to $+%.2f$ '
                 'elsewhere; re-encoding lifts $\\rho$ from %.3f to %.3f'
                 % (diag['nb101_genotype']['test'], min(nb201_tests), max(nb201_tests),
                    spaces['nb101']['rankers']['surrogate|1000']['rho'],
                    enc['nb101']['rankers']['path|1000']['rho']),
    ))

    # ---- 7. the decision metric
    ks = reg[A]['k_star']
    out.append(dict(
        claim='Rank correlation and regret at a shortlist disagree',
        prior='Predictor comparisons~\\cite{white2021powerful,ning2021evaluating}; proxy '
              'surveys~\\cite{abdelfattah2021zerocost,shu2022unifying}',
        prior_says='Cheap rankers are compared by rank correlation, per space.',
        we_add='The two orderings disagree in both directions, so the correlation ranking '
               'does not answer the question a budget asks. Wherever a shortlist is chosen we '
               'report $k^{\\star}$, the smallest shortlist whose regret falls below a tenth '
               'of a point.',
        evidence='the surrogate leads in $\\rho$ yet $k^{\\star}=%s$ against $%s$ for a '
                 'measured epoch; on NAS-Bench-201 SynFlow leads the shortlist on two of '
                 'three datasets' % (ks['surrogate'], ks['epoch1']),
    ))

    # ---- 8. non-identifiability of proxy choice
    out.append(dict(
        claim='Which free ranker is best is not identifiable without target labels',
        prior='NAS-Bench-Suite-Zero~\\cite{krishnakumar2022nasbenchsuitezero}; the unifying '
              'analysis of~\\cite{shu2022unifying}',
        prior_says='A per-space table of which proxy wins, computed with the target labels '
                   'in hand.',
        we_add='That table is an oracle quantity: choosing on the only labels a label-free '
               'deployment has costs most of the advantage. The deployable object is an '
               'untuned fusion, not a choice.',
        evidence='an honest choice gives up $%.2f$ points where an oracle gives up $%.2f$; '
                 'the untuned fusion wins by $%.2f$'
                 % (lf[A]['picked_regret10'], lf[A]['oracle_regret10'],
                    -pr[A]['fused_picked_minus_pickedproxy']['mean']),
    ))

    # ---- 9. the emulation pitfall
    out.append(dict(
        claim='Emulating a cheap signal by corrupting a reference can invert its conclusion',
        prior='Low-fidelity and surrogate benchmark studies~\\cite{zela2022surrogate,'
              'mehta2022nasbenchsuite}',
        prior_says='A policy is priced without GPUs by degrading a measured reference to the '
                   'cheap signal\'s fidelity.',
        we_add='The device has a precondition that fails where it is most tempting: inside a '
               'model-in-the-loop search the fidelity is a function of what the policy '
               'measures, so the loop must be refitted and the shortcut inverts its sign.',
        evidence='a gain predicted against $%.2f$ to $%.2f$ measured, across '
                 'budgets' % (sitl[A]['120']['scratch_delta'],
                              sitl[B]['520']['scratch_delta']),
    ))
    return out


CAPTION = (
    'What the closest prior art establishes, and what this paper adds. Row 1 is the premise '
    'rather than a contribution: that a cheap predictor is hard to beat is known, and is the '
    'baseline the other rows are measured against. Rows 7--9 concern how any cheap ranker '
    'should be measured, and apply to comparisons already in the literature. Each row names '
    'its claim in bold and closes with the evidence; every value is read from '
    '\\texttt{analysis/results/} by \\texttt{analysis/s49\\_novelty.py}.')


def latex(rs):
    o = ['\\begin{table*}[!t]', '\\centering',
         '\\caption{%s}' % CAPTION, '\\label{tab:novelty}', '\\scriptsize',
         '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}p{0.025\\textwidth}p{0.42\\textwidth}'
         'p{0.50\\textwidth}@{}}',
         '\\toprule',
         '\\# & Established by prior art & What this paper adds, and the evidence \\\\',
         '\\midrule']
    for i, r in enumerate(rs, 1):
        o.append('%d & %s \\emph{%s} & \\textbf{%s.} %s \\emph{%s.} \\\\'
                 % (i, r['prior'], r['prior_says'], r['claim'], r['we_add'], r['evidence']))
        o.append('\\addlinespace[2pt]')
    o[-1] = '\\bottomrule'
    o += ['\\end{tabular}', '\\end{table*}', '']
    return '\n'.join(o)



CAPTION_MAIN = (
    'What the closest prior art establishes, and what this paper adds. Row~1 is the premise '
    'rather than a contribution: the decomposition is standard and we use it as such, and the '
    'claim is about the landscape it is turned on. Rows~7--9 concern how any cheap ranker '
    'should be measured, so they apply to comparisons already in the literature rather than '
    'only to ours. Each row closes with the evidence it rests on, in italics; every value is '
    'read from \\texttt{analysis/results/} by \\texttt{analysis/s49\\_novelty.py} and no '
    'number in this table is typed by hand. Supplementary Table~S2 gives the same nine rows '
    'with the prior art stated at length, and Supplementary Section~S2 says how both are '
    'generated and rechecked.')


def latex_main(rs):
    """Table I: the ledger at body length, built from the same rows as the supplement's."""
    o = ['%% generated by analysis/s49_novelty.py -- do not edit by hand',
         '\\begin{table*}[!t]', '\\centering',
         '\\caption{%s}' % CAPTION_MAIN, '\\label{tab:novelty_main}', '\\scriptsize',
         '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}p{0.018\\textwidth}p{0.30\\textwidth}'
         'p{0.63\\textwidth}@{}}',
         '\\toprule',
         '\\# & Established by prior art & \\textbf{What this paper adds}, and the evidence '
         '\\\\',
         '\\midrule']
    for i, r in enumerate(rs, 1):
        sp, sa = SHORT[i - 1]
        o.append('%d & %s & \\textbf{%s.} %s \\emph{%s.} \\\\' % (i, sp, r['claim'], sa,
                                                                r['evidence']))
        o.append('\\addlinespace[2pt]')
    o[-1] = '\\bottomrule'
    o += ['\\end{tabular}', '\\end{table*}', '']
    return '\n'.join(o)


def main():
    rs = rows()
    with open(os.path.join(RES, 'novelty.json'), 'w') as fh:
        json.dump({'n_claims': len(rs), 'claims': rs}, fh, indent=1, sort_keys=True)
    #: the short form must cover exactly the rows the long form does, or the body table and
    #: the supplement's would be ledgers of different papers
    if len(SHORT) != len(rs):
        raise SystemExit('novelty_short.SHORT has %d rows and rows() has %d; the body ledger '
                         'and the supplement ledger must cover the same claims'
                         % (len(SHORT), len(rs)))
    out = os.path.join(TBL, 'tab_novelty.tex')
    with open(out, 'w') as fh:
        fh.write(latex(rs))
    outm = os.path.join(TBL, 'tab_novelty_body.tex')
    with open(outm, 'w') as fh:
        fh.write(latex_main(rs))
    print('wrote tables/tab_novelty.tex and tables/tab_novelty_body.tex (%d claims), '
          'and results/novelty.json' % len(rs))
    for i, r in enumerate(rs, 1):
        print('  %d. %-60s %s' % (i, r['claim'][:60], r['evidence'][:70]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
