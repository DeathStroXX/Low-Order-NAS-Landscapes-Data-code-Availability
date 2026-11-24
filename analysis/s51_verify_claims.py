"""Re-check every number quoted in main.tex and in supplementary.tex against analysis/results/.

A claim may be quoted in main.tex *or* in supplementary.tex, and the checker reports which,
while still requiring every computed value to format exactly into the quoted string. The guard
is simple and it is the one that matters: a number that drifts when an analysis is rerun cannot
survive into either document, and the same quantity quoted twice with two different values
still fails. Sentences get rewritten as a paper is revised; when that happens the computed
value is unchanged and only the quoted string moves, which is exactly what this checker exists
to track.

It is also the manuscript's contradiction check. It verifies that the novelty ledger quotes the
same values as the prose it summarizes -- a ledger that disagreed with the body would be worse
than none -- and that every reference main.tex makes into the supplement lands on the object it
names, which LaTeX cannot check because neither document can \\ref into the other.

Run after every analysis script and before submitting:
    python3 analysis/s51_verify_claims.py
"""
import json, os, re, sys

import common

_RAW = open(os.path.join(common.HERE, '..', 'main.tex')).read()
_SUP = open(os.path.join(common.HERE, '..', 'supplementary.tex')).read()
#: line breaks inside a sentence are irrelevant to a claim, so both sides are whitespace-folded
#: Tables are inlined into the documents by analysis/s70_assemble.py, between
#: \tablesource{tables/x.tex} and \tablesourceend{tables/x.tex} (both expand to nothing).
#: tables/ is still the source of truth, so a checker that used to resolve \input matches
#: either form and reads the same file. The markers are macros rather than comments because
#: this module strips full-line comments before it looks.
TABLE_REF = re.compile(r'(?:\\input|\\tablesource)\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}')
TABLE_BLOCK = re.compile(r'\\tablesource\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}'
                         r'.*?\\tablesourceend\{\1\}', re.S)

TEX = re.sub(r'\s+', ' ', _RAW)
SUP = re.sub(r'\s+', ' ', _SUP)
#: main.tex plus every table file it \input's, so a number quoted inside a generated caption
#: is checked too. Used for cross-reference checks, where the reference may live in a caption.
def _inputs(src):
    out = ''
    for rel in TABLE_REF.findall(src):
        rel = rel if rel.endswith('.tex') else rel + '.tex'
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', rel)
        if os.path.exists(path):
            out += '\n' + open(path).read()
    return out


TEXALL = re.sub(r'\s+', ' ', _RAW + _inputs(_RAW))
#: the supplement pulls in tables of its own -- the full novelty ledger among them -- and a
#: claim quoted inside one of those is quoted in the supplement, so the MOVED branch has to
#: see them too. Without this a row of a generated supplementary table reads as missing.
SUPALL = re.sub(r'\s+', ' ', _SUP + _inputs(_SUP))
OK, BAD, SKIPPED, MOVED = [], [], [], []
A, B = 'cifar10', 'cifar100'
NB = ['cifar10', 'cifar100', 'ImageNet16-120']


def J(n):
    return json.load(open(os.path.join(common.RES, n + '.json')))


def chk(label, literal, value=None, fmt='{:.3f}'):
    """`literal` must appear in main.tex; if `value` is given it must format into `literal`."""
    if value is not None:
        want = fmt(value) if callable(fmt) else fmt.format(value)
        if want not in re.sub(r'\s+', ' ', literal):
            BAD.append('%s: computed %s is NOT in the quoted string "%s"'
                       % (label, want, literal))
            return
    lit = re.sub(r'\s+', ' ', literal)
    if lit in TEX or lit in TEXALL:
        OK.append('%s: "%s"' % (label, lit[:60]))
    elif lit in SUP or lit in SUPALL:
        MOVED.append('%s: "%s"' % (label, lit[:60]))
    else:
        BAD.append('%s: NOT FOUND in main.tex or supplementary.tex: "%s"' % (label, lit))


def chks(label, literal, value=None, fmt='{:.3f}'):
    """As `chk`, but the literal must appear in supplementary.tex.

    The supplement carries the numbers the ten-page limit pushed out of the main paper, so it
    needs the same guard: a value quoted in one document and recomputed for the other is how
    the two drift apart.
    """
    if value is not None:
        want = fmt(value) if callable(fmt) else fmt.format(value)
        if want not in re.sub(r'\s+', ' ', literal):
            BAD.append('%s [supp]: computed %s is NOT in the quoted string "%s"'
                       % (label, want, literal))
            return
    lit = re.sub(r'\s+', ' ', literal)
    if lit in SUP:
        OK.append('%s [supp]: "%s"' % (label, lit[:60]))
    else:
        BAD.append('%s: NOT FOUND in supplementary.tex: "%s"' % (label, lit))


def zerocost():
    """Section V's comparison against the zero-cost proxy family, the two further spaces, the
    encoding failure and the task-family bound. Every number quoted in the main paper for these
    is recomputed here from results/zc_nb201.json, results/zc_spaces.json and
    results/encoding.json, so none of them can drift from the scripts that produced them."""
    z = J('zc_nb201')
    sp = J('zc_spaces')
    en = J('encoding')
    pd_ = z['per_dataset']
    lf = z['label_free']
    src = z['transfer_source_rule']['source']
    pr = z['paired']

    # the proxy release itself, and that we read it correctly
    chk('zc n proxies', r'publishes thirteen', len(z['proxies']),
        lambda v: 'thirteen' if v == 13 else str(v))
    chk('zc alignment pool', r'on the same pools and splits', None)

    # rank fidelity: transferred surrogate against the best proxy by rho
    QUOTE = ("$0.887$, $0.892$, $0.842$ against the best proxy's "
             "$0.792$, $0.805$, $0.780$")
    for i, d in enumerate(NB):
        chk('zc transfer rho ' + d, QUOTE,
            pd_[d]['transfer|' + src[d]]['rho'])
        chk('zc best proxy rho ' + d, QUOTE,
            pd_[d]['proxy|' + z['best_proxy'][d]]['rho'])

    # the decision metric: SynFlow against the transferred surrogate
    chk('zc synflow r10 c10',
        r'SynFlow gives up $0.21$ and $0.08$ points at a shortlist of',
        pd_[NB[0]]['proxy|synflow']['regret']['10'], '{:.2f}')
    chk('zc synflow r10 c100', r'$0.21$ and $0.08$ points at a shortlist of',
        pd_[NB[1]]['proxy|synflow']['regret']['10'], '{:.2f}')
    chk('zc transfer r10 c10', r"surrogate's $0.21$ and $0.12$",
        pd_[NB[0]]['transfer|' + src[NB[0]]]['regret']['10'], '{:.2f}')
    chk('zc transfer r10 c100', r"$0.21$ and $0.12$",
        pd_[NB[1]]['transfer|' + src[NB[1]]]['regret']['10'], '{:.2f}')
    chk('zc in16 proxy better', r'($0.89$ against $3.16$)',
        pd_[NB[2]]['proxy|synflow']['regret']['10'], '{:.2f}')
    chk('zc in16 transfer r10', r'against $3.16$',
        pd_[NB[2]]['transfer|' + src[NB[2]]]['regret']['10'], '{:.2f}')

    # the selection problem: honest pick against the oracle
    for i, d in enumerate(NB):
        chk('zc picked r10 ' + d,
            r'gives up $0.89$, $1.31$ and $0.89$ points where an oracle',
            lf[d]['picked_regret10'], '{:.2f}')
        chk('zc oracle r10 ' + d,
            [r'oracle choosing on the target gives up $0.21$, $0.08$ and $0.53$',
             r'$0.08$ and $0.53$', r'$0.53$'][i], lf[d]['oracle_regret10'], '{:.2f}')

    # the paired contrasts, with their intervals
    def interval(label, quoted, e, sign=-1.0):
        """A paired difference and both ends of its interval must all be quoted."""
        chk(label, quoted, sign * e['mean'], '{:.2f}')
        chk(label + ' lo', quoted, sign * e['ci'][0 if sign < 0 else 1], '{:.2f}')
        chk(label + ' hi', quoted, sign * e['ci'][1 if sign < 0 else 0], '{:.2f}')

    interval('zc paired transfer c10', r'$0.69$ $[0.61,0.77]$',
             pr[NB[0]]['transfer_minus_pickedproxy'])
    interval('zc paired transfer c100', r'$1.19$ $[1.06,1.36]$',
             pr[NB[1]]['transfer_minus_pickedproxy'])
    interval('zc paired transfer in16', r'$2.28$ $[1.75,2.85]$',
             pr[NB[2]]['transfer_minus_pickedproxy'], sign=1.0)
    interval('zc paired fused c10', r'$0.77$ $[0.67,0.85]$',
             pr[NB[0]]['fused_picked_minus_pickedproxy'])
    interval('zc paired fused c100', r'$1.11$ $[0.93,1.32]$',
             pr[NB[1]]['fused_picked_minus_pickedproxy'])
    chk('zc paired fused in16', r'($+0.45$ $[-0.07,1.03]$)',
        pr[NB[2]]['fused_picked_minus_pickedproxy']['mean'], '{:+.2f}')
    chk('zc paired fused in16 lo', r'$[-0.07,1.03]$',
        pr[NB[2]]['fused_picked_minus_pickedproxy']['ci'][0], '{:+.2f}')
    chk('zc paired fused in16 hi', r'$[-0.07,1.03]$',
        pr[NB[2]]['fused_picked_minus_pickedproxy']['ci'][1], '{:.2f}')
    chk('zc fused kstar', r'$k^{\star}$ to $20$ on both CIFAR datasets',
        lf[NB[0]]['fused_picked_kstar'], lambda v: '$k^{\\star}$ to $%d$' % v)

    # NAS-Bench-301 and NAS-Bench-101
    if 'nb301' in sp:
        r3 = sp['nb301']['rankers']
        b3 = sp['nb301']['best_proxy']
        chk('nb301 surrogate rho', r'$\rho=0.826$ against the best proxy',
            r3['surrogate|1000']['rho'])
        chk('nb301 best proxy rho', r"best proxy's $0.472$", r3['proxy|' + b3]['rho'])
        chk('nb301 surrogate r10', r'$0.16$ points of regret at ten against',
            r3['surrogate|1000']['regret']['10'], '{:.2f}')
        chk('nb301 proxy r10', r"$0.16$ points of regret at ten against the best shortlist's $0.66$" ,
            min(r3['proxy|' + m]['regret']['10'] for m in z['proxies']), '{:.2f}')
        chk('nb301 kstar', r'$k^{\star}=50$, which one of the thirteen',
            r3['surrogate|1000']['k_star'], lambda v: '$k^{\\star}=%d$' % v)
        # exactly one of the thirteen matches the surrogate's k-star here, which is what the
        # main paper now says; if a second one ever did, "which one of the thirteen matches"
        # would be wrong and this is what would catch it
        ties = [m for m in z['proxies']
                if r3['proxy|' + m]['k_star'] == r3['surrogate|1000']['k_star']]
        if len(ties) != 1:
            BAD.append('nb301 k-star: %d proxies match the surrogate, but the main paper says '
                       'one' % len(ties))
        else:
            OK.append('nb301 k-star: exactly one proxy matches the surrogate')
        chk('nb301 genes', r'28 varying genes', None)
    if 'nb101' in sp:
        r1 = sp['nb101']['rankers']
        chk('nb101 surrogate rho',
            r'the same comparison locates the boundary: $\rho=0.499$',
            r1['surrogate|1000']['rho'])
        chk('nb101 zen rho', r"Zen's $0.594$", r1['proxy|zen']['rho'])

    # the repair and the diagnostic
    e = en['nb101']['rankers']
    chk('nb101 path rho', r'lifts $\rho$ to $0.692$', e['path|1000']['rho'])
    chk('nb101 genotype r10', r'regret at ten from $0.90$ to $0.40$',
        e['genotype|1000']['regret']['10'], '{:.2f}')
    chk('nb101 path r10', r'from $0.90$ to $0.40$',
        e['path|1000']['regret']['10'], '{:.2f}')
    chk('nb101 path kstar', r'$k^{\star}$ falling from above 500 to 100',
        e['path|1000']['k_star'], lambda v: 'to %d' % v)
    chk('diag r2 nb101', r'held-out $R^{2}$ is $-0.10$ there',
        en['diagnostic']['nb101_genotype']['test'], '{:+.2f}')
    worst = min(en['diagnostic'][k]['test'] for k in
                ['nb201_cifar10_genotype', 'nb201_cifar100_genotype',
                 'nb201_ImageNet16-120_genotype', 'nb301_genotype'])
    best = max(en['diagnostic'][k]['test'] for k in
               ['nb201_cifar10_genotype', 'nb201_cifar100_genotype',
                'nb201_ImageNet16-120_genotype', 'nb301_genotype'])
    chk('diag r2 range low', r'against $+0.58$ to $+0.84$', worst, '{:+.2f}')
    chk('diag r2 range high', r'$+0.58$ to $+0.84$', best, '{:+.2f}')
    chks('diag rejected supp', r'$+0.05$ for the encoding that fails',
         en['rejected_diagnostic_label_gain']['nb101_genotype']['gain'], '{:+.2f}')
    chks('nb101 paths supp', r'121 distinct paths occur', en['nb101']['n_paths'],
         lambda v: '%d distinct paths' % v)

    # the task-family bound
    ex = z['extra_tasks']
    chk('svhn transfer', r'ranks the unseen natural-image dataset SVHN at $\rho=0.823$',
        ex['svhn']['transfer|' + ex['svhn']['best_transfer']]['rho'])
    chk('svhn proxy', r'above the best proxy there ($0.792$)',
        ex['svhn']['proxy|' + ex['svhn']['best_proxy']]['rho'])
    chk('scifar transfer', r'$0.04$ on spherical',
        ex['scifar100']['transfer|' + ex['scifar100']['best_transfer']]['rho'], '{:.2f}')
    chk('ninapro transfer', r'$0.12$ on NinaPro',
        ex['ninapro']['transfer|' + ex['ninapro']['best_transfer']]['rho'], '{:.2f}')
    chk('scifar proxy', r'better ($0.19$ and $0.32$)',
        ex['scifar100']['proxy|' + ex['scifar100']['best_proxy']]['rho'], '{:.2f}')
    chk('ninapro proxy', r'better ($0.19$ and $0.32$)',
        ex['ninapro']['proxy|' + ex['ninapro']['best_proxy']]['rho'], '{:.2f}')

    # the pre-registration must agree with the two effect sizes the paper predicts
    cf = J('confirm')
    chk('prereg c1 c10', r'$+4.28$ and $+7.95$ points for the budget recipe',
        cf['claims']['C1']['by_dataset']['cifar10']['predicted'], '{:.2f}')
    chk('prereg c2 c10', r'$+4.18$ and $+4.76$ for the seeded',
        cf['claims']['C2']['by_dataset']['cifar10']['predicted'], '{:.2f}')
    chks('prereg supp c1', r'($+4.28$ and $+7.95$ points for the budget recipe',
         cf['claims']['C1']['by_dataset']['cifar10']['predicted'], '{:+.2f}')
    chks('prereg supp c2', r'$+4.18$ and $+4.76$ for the seeded',
         cf['claims']['C2']['by_dataset']['cifar10']['predicted'], '{:+.2f}')


def crossrefs():
    """Every `Supplementary Table~Sn` / `Section~Sn` / `Fig.~Sn` in main.tex must name a label
    that supplementary.tex actually assigns that number to.

    The supplement renumbers whenever a section is inserted, so a reference written by hand
    goes stale silently. This check reads supplementary.aux -- the numbers LaTeX really
    assigned -- and fails on any reference to a number that does not exist, plus on the
    handful of references whose intended target is pinned by label below.
    """
    import collections
    auxp = os.path.join(common.HERE, '..', 'supplementary.aux')
    if not os.path.exists(auxp):
        SKIPPED.append('cross-reference checks: supplementary.aux is not present, so the six '
                       'checks that confirm every "Supplementary Table~Sn" in main.tex still '
                       'points where it should could not run. Build the supplement first '
                       '(latexmk -pdf supplementary.tex) and rerun.')
        return
    lab = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{(S?\d+)\}', open(auxp).read()):
        lab[m.group(1)] = m.group(2)
    have = collections.defaultdict(set)
    for k, v in lab.items():
        kind = ('sec' if k.startswith('sec:') else 'tab' if k.startswith('tab:')
                else 'fig' if k.startswith('fig:') else 'lst')
        have[kind].add(v)
    KIND = {'Section': 'sec', 'Table': 'tab', 'Fig.': 'fig', 'Listing': 'lst'}
    dangling = []
    for m in re.finditer(r'Supplementary (Section|Table|Fig\.|Listing)s?~?(S\d+)'
                         r'(?:--(S\d+))?', TEXALL):
        for g in (m.group(2), m.group(3)):
            if g and g not in have[KIND[m.group(1)]]:
                dangling.append(m.group(0))
    for m in re.finditer(r'(?<!Supplementary )(Table|Fig\.)~(S\d+)', TEXALL):
        if m.group(2) not in have[{'Table': 'tab', 'Fig.': 'fig'}[m.group(1)]]:
            dangling.append(m.group(0))
    if dangling:
        BAD.append('crossrefs: main.tex points at supplement numbers that do not exist: %s'
                   % sorted(set(dangling)))
    else:
        OK.append('crossrefs: every supplement reference in main.tex resolves')

    #: Every reference main.tex makes INTO the supplement, pinned to the object it must land
    #: on, because LaTeX cannot \ref across the two documents and so cannot catch a drift.
    #: References *within* the supplement use \ref and need no pin; s52_consistency.py checks
    #: the same pointers in both directions and is the fuller instrument.
    PINNED = [('the novelty ledger', 'sec:sm_novelty', 'Supplementary Section~%s'),
              ('the record and the surrogate', 'sec:sm_fractal', 'Supplementary Section~%s'),
              ('the rejected diagnostic', 'sec:sm_spaces', 'Supplementary Section~%s'),
              ('the dose-response', 'sec:sm_targets', 'Supplementary Section~%s'),
              ('the measurement findings', 'sec:sm_measure', 'Supplementary Section~%s'),
              ('the per-proxy detail', 'tab:zc_sm', 'Supplementary Tables~%s'),
              ('the two further spaces', 'tab:zc_spaces', 'and~%s'),
              ('the controls figure', 'fig:sm_controls', 'Supplementary Fig.~%s')]
    for label, key, pattern in PINNED:
        if key not in lab:
            continue
        want = pattern % lab[key] if '%s' in pattern else pattern
        if want in TEXALL:
            OK.append('crossref %s: "%s"' % (label, want))
        else:
            BAD.append('crossref %s: main.tex (or a table it inputs) should contain "%s" '
                       '(label %s is %s)' % (label, want, key, lab[key]))


def absent(label, literal):
    """A string that must NOT appear -- a claim the revision retired."""
    lit = re.sub(r'\s+', ' ', literal)
    if lit in TEX:
        BAD.append('%s: should no longer appear in main.tex: "%s"' % (label, lit))
    else:
        OK.append('%s: correctly absent' % label)


#: English for the small counts the manuscript spells out, so a count that moves when an
#: analysis is rerun fails here rather than surviving as a word the value checker cannot see.
_WORDS = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
          7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
          13: 'thirteen'}


def spectrum_claims():
    """Every number the landscape sections quote, recomputed from spectrum.json.

    These are the claims the paper now leads with, so they get the same guard as the rest --
    and two of them need it more than most, because they are the ones that corrected an
    earlier recommendation. If the recalibrated spectrum stopped being flat across tasks, or
    recalibration stopped beating transfer, the argument of Section VI would be wrong and this
    function is what would say so.
    """
    if not os.path.exists(os.path.join(common.RES, 'spectrum.json')):
        BAD.append('results/spectrum.json is missing -- run python3 analysis/s53_spectrum.py')
        return
    S = J('spectrum')
    sp, rc, nl = S['spectrum'], S['spectrum_recalibrated'], S['null']
    D = S['datasets']

    # the exact decomposition rests on the space being a complete factorial
    chk('spectrum: complete factorial', r'all $15{,}625$ architectures', S['n_arch'],
        lambda v: '%d{,}%03d' % (v // 1000, v % 1000))

    # the cumulative shares, quoted in the abstract, the contributions and Section IV
    for d, lo2 in zip(D, [sp[x]['cumulative']['2'] for x in D]):
        chk('spectrum: order<=2 %s' % d,
            r'carries $67.9\%$, $78.1\%$ and $88.2\%$', 100 * lo2, '{:.1f}')
    for d, lo3 in zip(D, [sp[x]['cumulative']['3'] for x in D]):
        chk('spectrum: order<=3 %s' % d,
            r'order $\le 3$ carries $93.4\%$, $95.6\%$ and $96.7\%$', 100 * lo3, '{:.1f}')

    # the white-noise null, which is what makes the tail evidence of structure
    chk('spectrum: null at order 6', r'$26.2\%$ of its variance at order six',
        100 * nl['6'], '{:.1f}')
    chk('spectrum: null at order 1', r'$0.15\%$ at order one', 100 * nl['1'], '{:.2f}')
    chk('spectrum: measured order 6', r'measured order-six share is $0.14\%$ on CIFAR-10',
        100 * sp['cifar10']['share']['6'], '{:.2f}')
    chk('spectrum: null ratio', r'a factor of $192$ below the null',
        nl['6'] / sp['cifar10']['share']['6'], '{:.0f}')

    # the recalibrated spectrum: the three tasks must actually coincide
    chk('spectrum: recalibrated',
        r'the three spectra nearly coincide at $89.5\%$, $88.4\%$ and $88.0\%$',
        100 * rc['cifar10']['cumulative']['2'], '{:.1f}')
    spread = max(rc[d]['cumulative']['2'] for d in D) - min(rc[d]['cumulative']['2'] for d in D)
    if spread > 0.03:
        BAD.append('the recalibrated spectra differ by %.1f points across tasks, so the '
                   'claim that they coincide no longer holds' % (100 * spread))
    else:
        OK.append('recalibrated spectra coincide across tasks (spread %.1f points)'
                  % (100 * spread))

    # the compression statistics that identify the mechanism
    C = S['compression']
    chk('compression: top decile', r'occupies $2.1\%$ of the accuracy range',
        100 * C['cifar10']['top_decile_spread'], '{:.1f}')
    chk('compression: c100 decile', r'against $5.3\%$ on CIFAR-100',
        100 * C['cifar100']['top_decile_spread'], '{:.1f}')
    chk('compression: in16 decile', r'$9.4\%$ on ImageNet16-120',
        100 * C['ImageNet16-120']['top_decile_spread'], '{:.1f}')
    chk('compression: skew', r'the skewness runs $-4.79$, $-3.19$, $-1.35$',
        C['cifar10']['skew'], '{:.2f}')

    # the correction itself: recalibration must beat transfer everywhere it is claimed to
    R = S['recalibration']
    c10 = R['cifar10']['budgets']['1000']
    chk('recal: c10 rho', r'it reaches $\rho=0.935$ against the transferred',
        c10['own_recal']['rho'], '{:.3f}')
    chk('recal: c10 transfer rho', r"transferred prior's $0.879$", c10['transfer_raw']['rho'],
        '{:.3f}')
    chk('recal: c10 raw rho', r"uncorrected fit's $0.682$", c10['own_raw']['rho'], '{:.3f}')
    chk('recal: c10 regret', r'gives up $0.05$ points at a shortlist', c10['own_recal']['regret10'],
        '{:.2f}')
    losses = [(d, n) for d in D for n in ('100', '250', '500', '1000')
              if not (R[d]['budgets'][n]['own_recal']['rho']
                      > R[d]['budgets'][n]['transfer_raw']['rho']
                      and R[d]['budgets'][n]['own_recal']['regret10']
                      < R[d]['budgets'][n]['transfer_raw']['regret10'])]
    if losses:
        BAD.append('recalibration does not beat transfer at %s, but Section VI claims it wins '
                   'at every budget on both metrics' % losses)
    else:
        OK.append('recalibration beats transfer at every budget on both metrics')

    # and the control: in an uncompressed space the correction must not reproduce the gain.
    # The paper's claim concerns both rank fidelity and regret, so both are checked, and the
    # quoted magnitudes are checked too.
    if 'fractalnet' in S:
        F = S['fractalnet']
        chk('control: skew (protocol)', r'spread out (skewness $-0.41$ and $+0.89$)',
            (F['cifar10']['skew'], F['cifar100']['skew']), lambda v: '$%+.2f$ and $%+.2f$' % v)
        chk('control: top decile', r'the top decile occupies $11\%$ and $43\%$',
            (100 * F['cifar10']['top_decile_spread'], 100 * F['cifar100']['top_decile_spread']),
            lambda v: '$%.0f\\%%$ and $%.0f\\%%$' % v)
        chk('control: top decile (Fig. 1)', r'and the uncompressed control $11.1\%$',
            100 * F['cifar10']['top_decile_spread'], '{:.1f}')
        chk('control: skew', r'have skewness $-0.41$ and $+0.89$', F['cifar10']['skew'],
            '{:.2f}')
        drho = max(abs(b['own_recal']['rho'] - b['own_raw']['rho'])
                   for v in F.values() for b in v['budgets'].values())
        if drho > 0.025:
            BAD.append('recalibration moves rank fidelity by %.3f in the uncompressed control '
                       'space, which is too large for the paper to call it no such gain'
                       % drho)
        else:
            OK.append('control: recalibration moves rank fidelity by at most %.3f' % drho)
        # the paper says so explicitly for CIFAR-10, and that regret gets worse on CIFAR-100
        c10 = F.get('cifar10', {}).get('budgets', {})
        if c10:
            d10 = max(abs(b['own_recal']['rho'] - b['own_raw']['rho']) for b in c10.values())
            chk('control: c10 rho move', r'by at most $0.004$ on CIFAR-10', d10,
                lambda v: '%.3f' % (round(v, 3) + (0.001 if round(v, 3) < v else 0)))
            #: the three regret moves are quoted, so the claim is checked rather than asserted
            d10r = sorted(b['own_recal']['regret10'] - b['own_raw']['regret10']
                          for b in c10.values())
            chk('control: c10 regret range', r'it lowers regret by $0.12$ to $0.39$ points',
                (-max(d10r), -min(d10r)), lambda v: '$%.2f$ to $%.2f$ points' % v)
            # "roughly halving it at 1,000 labels", and lower at every budget
            ratio = c10['1000']['own_recal']['regret10'] / c10['1000']['own_raw']['regret10']
            if not 0.4 <= ratio <= 0.6 or max(d10r) > 0:
                BAD.append('control: CIFAR-10 regret ratio at 1,000 labels %.2f, worst move %+.2f; '
                           'the text says it falls everywhere and roughly halves' % (ratio, max(d10r)))
            else:
                OK.append('control: CIFAR-10 regret falls everywhere, ratio %.2f at 1,000' % ratio)
        c100 = F.get('cifar100', {}).get('budgets', {})
        if c100:
            worse = sorted(b['own_recal']['regret10'] - b['own_raw']['regret10']
                           for b in c100.values())
            chk('control: c100 regret worse', r'by $0.02$ to $1.21$ points',
                (worse[0], worse[-1]), lambda v: '$%.2f$ to $%.2f$ points' % v)
            if worse[0] <= 0:
                BAD.append('control: recalibration does not worsen CIFAR-100 regret at every budget')
            d100 = max(abs(b['own_recal']['rho'] - b['own_raw']['rho']) for b in c100.values())
            chk('control: c100 rho move', r'and $0.019$ on CIFAR-100, and', d100,
                lambda v: '%.3f' % (round(v, 3) + (0.001 if round(v, 3) < v else 0)))

        # the logit check the Limitations now rests on
        if 'spectrum_logit' in S:
            spr = {}
            for lab, key in (('raw', 'spectrum'), ('logit', 'spectrum_logit'),
                             ('rank', 'spectrum_recalibrated')):
                v = [S[key][d]['cumulative']['2'] for d in D]
                spr[lab] = 100 * (max(v) - min(v))
            chk('logit: raw spread', r'where accuracy has $20.3$', spr['raw'], '{:.1f}')
            # the logit and rank spreads are not quoted in either document (they are in
            # results/spectrum.json)


def search_claims():
    """Section VIII-A quotes the replayed search exactly, so every figure is re-read here.

    The replay is a lookup in the published table rather than a measurement, which makes the
    numbers reproducible to the last digit and therefore worth pinning: a rerun that moved one
    of them would mean the protocol had changed, not that a network had trained differently.
    """
    se = J('nb201_search')
    res, grid, led = se['results'], se['grid'], se['ledger']

    def at(ds, alg, n, key='acc_mean'):
        return res[ds][alg][str(n)][key]

    # --- the ledger: the headline is a count over every cell, not one chosen budget
    chk('ledger cells', r'three datasets by seven budgets, $21$ in all',
        led['evolution']['n_cells'], '{:d}')
    chk('ledger vs random', r'the procedure is ahead in $21$ of $21$',
        led['random']['ahead'], '{:d}')
    for key, lit in (('ahead', r'it is ahead in $14$, tied in $5$ and behind in $2$'),
                     ('tied', r'it is ahead in $14$, tied in $5$ and behind in $2$'),
                     ('behind', r'it is ahead in $14$, tied in $5$ and behind in $2$')):
        chk('ledger vs evolution (%s)' % key, lit, led['evolution'][key], '{:d}')
    chk('abstract: evolution tally', r'regularized evolution in $14$ (tied in $5$, behind in $2$)',
        (led['evolution']['ahead'], led['evolution']['tied'], led['evolution']['behind']),
        lambda v: 'in $%d$ (tied in $%d$, behind in $%d$)' % v)
    for key in ('ahead', 'tied', 'behind'):
        chk('ledger vs local (%s)' % key,
            r'Against local search it is ahead in $18$, tied in $2$ and behind in $1$',
            led['local'][key], '{:d}')
    #: the two losing cells are named in the body, so the body cannot keep that sentence if
    #: a rerun moves the loss somewhere else
    behind = {(d, int(n)) for d, n in led['evolution']['behind_at']}
    if behind != {('cifar100', 100), ('cifar100', 200)}:
        BAD.append('ledger: the body names CIFAR-100 at N=100 and N=200 as the two cells '
                   'behind regularized evolution; the result file says %s' % sorted(behind))
    if led['recipe_leads_evolution_everywhere_at'][:1] != [20]:
        BAD.append('ledger: the body claims N=20 is the smallest budget at which the recipe '
                   'leads regularized evolution on all three datasets; the result file says %s'
                   % led['recipe_leads_evolution_everywhere_at'])

    # --- the small-budget contrast (absolute values: the rows of Table IV)
    chk('recipe at 20', r'$N{=}20$ & 93.50\,$\pm$\,0.04 & 71.18\,$\pm$\,0.10 & 45.04',
        at('cifar10', 'recipe', 20), '{:.2f}')
    chk('recipe at 20 (c100)', r'$N{=}20$ & 93.50\,$\pm$\,0.04 & 71.18\,$\pm$\,0.10 & 45.04',
        at('cifar100', 'recipe', 20), '{:.2f}')
    chk('recipe at 20 (in16)', r'$N{=}20$ & 93.50\,$\pm$\,0.04 & 71.18\,$\pm$\,0.10 & 45.04',
        at('ImageNet16-120', 'recipe', 20), '{:.2f}')
    _lit20 = (r'by $+0.13$ $[0.07,0.19]$, $+0.47$ $[0.33,0.61]$ and $+0.31$ $[0.15,0.47]$ '
              r'points')
    for ds in NB:
        c = res[ds]['recipe']['20']['vs_evolution']
        chk('recipe vs evolution at 20 (%s)' % ds, _lit20, c['diff'], '{:.2f}')
        chk('recipe vs evolution at 20, lo (%s)' % ds, _lit20, abs(c['ci'][0]), '{:.2f}')
        chk('recipe vs evolution at 20, hi (%s)' % ds, _lit20, abs(c['ci'][1]), '{:.2f}')
        if c['verdict'] != 'ahead':
            BAD.append('the body says the recipe leads regularized evolution on every dataset '
                       'at N=20; on %s the interval says "%s"' % (ds, c['verdict']))
    for ds in NB:
        chk('evolution at 100 (%s)' % ds,
            r'$N{=}100$ & 94.03\,$\pm$\,0.03 & 72.76\,$\pm$\,0.05 & 46.62', at(ds, 'evolution', 100), '{:.2f}')

    # --- the two tests, and how often they part
    ct = se['contrast_tests']
    chk('test disagreements', r'here they do in $8$ of $189$ contrasts',
        ct['n_disagreements'], '{:d}')
    chk('total contrasts', r'$8$ of $189$ contrasts', ct['n_contrasts'], '{:d}')

    chk('recipe at 100',
        r'$N{=}100$ & 94.05\,$\pm$\,0.03 & 72.62\,$\pm$\,0.05 & 46.67',
        at('cifar10', 'recipe', 100), '{:.2f}')
    chk('recipe at 100 (c100)', r'$N{=}100$ & 94.05\,$\pm$\,0.03 & 72.62\,$\pm$\,0.05 & 46.67',
        at('cifar100', 'recipe', 100), '{:.2f}')
    chk('recipe at 100 (in16)', r'$N{=}100$ & 94.05\,$\pm$\,0.03 & 72.62\,$\pm$\,0.05 & 46.67',
        at('ImageNet16-120', 'recipe', 100), '{:.2f}')
    chk('random at 100', r'$N{=}100$ & 93.74\,$\pm$\,0.03 & 71.59\,$\pm$\,0.06 & 45.68',
        at('cifar10', 'random', 100), '{:.2f}')
    chk('random at 100 (c100)', r'$N{=}100$ & 93.74\,$\pm$\,0.03 & 71.59\,$\pm$\,0.06 & 45.68',
        at('cifar100', 'random', 100), '{:.2f}')
    chk('random at 100 (in16)', r'$N{=}100$ & 93.74\,$\pm$\,0.03 & 71.59\,$\pm$\,0.06 & 45.68',
        at('ImageNet16-120', 'random', 100), '{:.2f}')
    for ds, lit in (('cifar10', 0), ('cifar100', 1), ('ImageNet16-120', 2)):
        gain = at(ds, 'recipe', 100) - at(ds, 'random', 100)
        chk('recipe gain over random (%s)' % ds,
            r'$+0.31$ $[0.27,0.35]$, $+1.03$ $[0.94,1.11]$ and $+1.00$ $[0.93,1.06]$ points',
            gain, '{:.2f}')
        share = 100.0 * gain / at(ds, 'random', 100, 'regret_mean')
        chk('share of random regret closed (%s)' % ds, r'which is $33$, $62$ and $60\%$',
            share, '{:.0f}')
    chk('validation oracle on c10',
        r'the $94.36$ that is the best any',
        grid['cifar10']['best_by_search_signal'], '{:.2f}')
    chk('grid optimum c10', r'a test optimum of $94.68$',
        grid['cifar10']['optimum'], '{:.2f}')
    chk('recipe cost at 100', r'$63$ against $51$ GPU-hours at $N=100$',
        at('cifar10', 'recipe', 100, 'gpu_hours_median'), '{:.0f}')
    chk('random cost at 100', r'$63$ against $51$ GPU-hours',
        at('cifar10', 'random', 100, 'gpu_hours_median'), '{:.0f}')
    if not se['checks']['no_algorithm_exceeds_grid_optimum']:
        BAD.append('replayed search: an algorithm beats the grid optimum, which is impossible')


def counts_and_superlatives():
    """Guard the claims that are counts or universals rather than measured quantities.

    Every failure this class has ever turned up had the same shape: a count
    or a superlative written as a word -- "four of the thirteen", "four discrete spaces",
    "neither ingredient", "all three" -- that the ordinary value checker steps straight over,
    because there is no decimal in it to compare. Each one below is recomputed from
    `analysis/results/` and compared against the word the manuscript uses.
    """
    zc, sp = J('zc_nb201'), J('zc_spaces')

    # how many of the thirteen proxies give a better NAS-Bench-101 shortlist than the ranker
    r101 = sp['nb101']['rankers']
    base = r101['surrogate|1000']['regret']['10']
    beat = sum(1 for k, v in r101.items()
               if k.startswith('proxy|') and v['regret']['10'] < base)
    chk('nb101 proxies beating the shortlist',
        r'a worse shortlist than %s of the thirteen' % _WORDS[beat])

    # how many discrete spaces the proxy release actually covers
    n_spaces = 1 + len([k for k in ('nb101', 'nb301') if k in sp])
    chk('number of spaces the proxy release covers (abstract)',
        r'for every architecture of %s spaces' % _WORDS[n_spaces])
    chks('number of spaces the proxy release covers (supplement)',
         r'%s discrete spaces' % _WORDS[n_spaces])
    chk('number of spaces the proxy release covers (conclusion)',
        r'architecture of %s spaces' % _WORDS[n_spaces])

    # the fusion reaches k*=20 on both CIFAR datasets; on how many is that better than BOTH
    # ingredients? The manuscript names the datasets, so the count has to match.
    improves = [d for d in (A, B)
                if zc['label_free'][d]['fused_picked_kstar']
                < min(zc['label_free'][d]['transfer_kstar'],
                      zc['label_free'][d]['picked_kstar'])]
    if improves != [A]:
        BAD.append('k-star claim: the fusion beats both ingredients on %s, but the manuscript '
                   'says CIFAR-10 alone' % (improves or 'neither'))
    else:
        OK.append('k-star claim: the fusion beats both ingredients on CIFAR-10 alone')
    chk('kstar fusion scope', r'on both CIFAR datasets, which on CIFAR-10 neither ingredient')

    # the ImageNet16-120 prior is negative on spherical CIFAR-100, so the 0.04 quoted for that
    # task is the best of the three priors and not that one
    ex = zc['extra_tasks']
    chk('spherical CIFAR-100, ImageNet prior', r'on spherical CIFAR-100, at $-0.05$',
        ex['scifar100']['transfer|ImageNet16-120']['rho'], '{:.2f}')
    chk('spherical CIFAR-100, best prior', r'reaches $0.04$ on spherical CIFAR-100',
        ex['scifar100']['transfer|' + ex['scifar100']['best_transfer']]['rho'], '{:.2f}')

    # on how many NAS-Bench-201 datasets does SynFlow give up less at ten than the transferred
    # surrogate? The manuscript says two of the three, with the third a tie to two decimals.
    wins, ties = 0, 0
    for d in NB:
        src = zc['label_free'][d]['best_transfer_source']
        sy = zc['per_dataset'][d]['proxy|synflow']['regret']['10']
        tr = zc['per_dataset'][d]['transfer|' + src]['regret']['10']
        if sy < tr:
            wins += 1
        elif round(sy, 2) == round(tr, 2):
            ties += 1
    if (wins, ties) != (2, 1):
        BAD.append('SynFlow shortlist claim: it leads on %d of 3 and ties on %d, but the '
                   'manuscript says two and one' % (wins, ties))
    else:
        OK.append('SynFlow leads the shortlist on two of three, ties on the third')
    chk('synflow shortlist scope',
        r'SynFlow gives up less at a shortlist of ten on two of three datasets')


def novelty_and_decision():
    """The novelty ledger and Fig. 1 must agree with the prose they summarise.

    Both are generated (s49_novelty.py, s50_decision.py) from the same results files as the
    body, so the risk is not arithmetic but staleness: a table regenerated from an older
    results/ directory, or a figure left behind by a rerun. Each value below is therefore
    checked twice -- that the generated artifact holds it, and that the manuscript quotes the
    same number somewhere -- which is what makes a disagreement between the ledger and the body
    impossible to ship.
    """
    nov_p = os.path.join(common.RES, 'novelty.json')
    dec_p = os.path.join(common.RES, 'decision.json')
    tbl_p = os.path.join(common.HERE, '..', 'tables', 'tab_novelty.tex')
    for path, how in ((nov_p, 's49_novelty.py'), (dec_p, 's50_decision.py'),
                      (tbl_p, 's49_novelty.py')):
        if not os.path.exists(path):
            BAD.append('artifact missing: %s -- run python3 analysis/%s'
                       % (os.path.relpath(path, os.path.join(common.HERE, '..')), how))
            return

    nov = json.load(open(nov_p))
    dec = json.load(open(dec_p))
    tbl = re.sub(r'\s+', ' ', open(tbl_p).read())

    # -- the ledger has one row per claim the paper makes, and the premise row is row 1
    if nov['n_claims'] != 9:
        BAD.append('novelty ledger: %d claims, expected 9' % nov['n_claims'])
    # row 1 is the exact spectrum, and it must concede that the apparatus is standard
    if 'prior art' not in nov['claims'][0]['we_add']:
        BAD.append('novelty ledger: row 1 must concede that the apparatus is prior art')

    # -- every row's evidence string must actually be in the generated table
    for i, c in enumerate(nov['claims'], 1):
        ev = re.sub(r'\s+', ' ', c['evidence'])
        if ev not in tbl:
            BAD.append('novelty row %d: evidence "%s" is not in tables/tab_novelty.tex '
                       '(regenerate with s49_novelty.py)' % (i, ev[:60]))
        else:
            OK.append('novelty row %d evidence in Table I' % i)

    # -- the ledger and Fig. 1 must not disagree with the body about a shared quantity
    zc, enc, spaces = J('zc_nb201'), J('encoding'), J('zc_spaces')
    rec, sitl, reg, nb = J('recipe'), J('sitl'), J('regret'), J('nb201')
    shared = [
        ('transfer rho out of the hardest dataset', r'$0.887$',
         zc['label_free'][A]['transfer_rho'], '{:.3f}'),
        ('own-label baseline it beats', r'$0.887$ against $0.737$',
         nb['fidelity'][A]['1000']['rho'], '{:.3f}'),
        ('held-out R2 at the boundary', r'$-0.10$ there',
         enc['diagnostic']['nb101_genotype']['test'], '{:.2f}'),
        ('path re-encoding', r'lifts $\rho$ to $0.692$',
         enc['nb101']['rankers']['path|1000']['rho'], '{:.3f}'),
        ('NB-101 genotype rho', r'$\rho=0.499$ against',
         spaces['nb101']['rankers']['surrogate|1000']['rho'], '{:.3f}'),
        ('composed recipe gain', r'the full recipe returns $+2.74$ and $+6.13$ points',
         rec[A]['cells']['C_screen_halve']['acc'] - rec[A]['cells']['A_train100']['acc'],
         '{:.2f}'),
        ('model-in-the-loop gain', r'gains $\mathbf{+4.62}$ and $\mathbf{+4.42}$ points',
         sitl[A]['120']['transfer_delta'], '{:.2f}'),
        ('the loop without a prior', r'$6.64$ and $7.68$ points',
         abs(sitl[A]['120']['scratch_delta']), '{:.2f}'),
    ]
    for label, literal, value, fmt in shared:
        chk('shared: ' + label, literal, value, fmt)

    # -- Fig. 1 is drawn from the same numbers, so its JSON must match them too
    for key, value, fmt in [('r2_bad', enc['diagnostic']['nb101_genotype']['test'], '{:.2f}'),
                            ('rho_path', enc['nb101']['rankers']['path|1000']['rho'],
                             '{:.3f}'),
                            ('kstar', reg[A]['k_star']['surrogate'], '{}'),
                            ('sitl_a', sitl[A]['120']['transfer_delta'], '{:.2f}'),
                            ('recipe_a', rec[A]['cells']['C_screen_halve']['acc']
                             - rec[A]['cells']['A_train100']['acc'], '{:.2f}')]:
        if fmt.format(dec[key]) != fmt.format(value):
            BAD.append('Fig. 1 is stale: decision.json[%r] = %s, results say %s '
                       '(rerun s50_decision.py)'
                       % (key, fmt.format(dec[key]), fmt.format(value)))
        else:
            OK.append('Fig. 1 %s agrees with results/' % key)

    # -- structure: the sections and floats the argument depends on must be present
    for label, literal in [
            ('the measurement subsection exists',
             r'\subsection{Reporting Regret Beside Rank Correlation}'),
            ('the spectrum section exists', r'\section{Exact Spectra of the Accuracy Landscape}'),
            ('the scale section exists',
             r'\section{Accuracy Scale and Transfer}'),
            ('recal table is carried', r'\tablesource{tables/tab_recal.tex}'),
            ('spectrum table is carried', r'\tablesource{tables/tab_spectrum.tex}'),
            ('recalibration table is carried', r'\tablesource{tables/tab_recal.tex}'),
            ('spectrum figure is included', r'figures/fig_spectrum_panels.pdf'),
            ('controls figure is included', r'figures/fig_controls.pdf'),
            ('Table II is referenced', r'Table~\ref{tab:recal}'),
            ('Fig. 1 is referenced', r'Fig.~\ref{fig:spectrum}')]:
        if re.sub(r'\s+', ' ', literal) in TEXALL:
            OK.append('structure: ' + label)
        else:
            BAD.append('structure: %s -- "%s" not found in main.tex' % (label, literal))

    # -- the four measurement findings must each be stated, not merely alluded to
    for label, literal in [
            ('metric disagreement', r'Rank correlation and regret at a shortlist disagree'),
            ('residual information',
             r'a measured epoch belongs in a second stage on a shortlist'),
            ('non-identifiability',
             r'Which free ranker is best is not identifiable without target labels'),
            ('emulation pitfall',
             r'emulating a cheap signal by corrupting a reference to')]:
        chk('finding: ' + label, literal)


def realization_dimension_and_protocol():
    """The realization, dimension and protocol numbers, each against the file that made it.

    Nothing here is new evidence: it is the same enumerated grids read in the three ways the
    body now reports them -- how much of the exact degree-2 optimum each cheap ranker gets,
    what determining such a model costs, and what the single-column search protocol is worth.
    The checks exist because a quoted percentage is exactly the kind of number that drifts
    silently when a result file is recomputed.
    """
    re_, dim, mo, sr = J('realization'), J('dimension'), J('monotone'), J('nb201_search')

    # ---- what each ranker realizes of the exact degree-2 optimum
    row = {r['key']: r for r in re_['rows']}
    chk('realization: honest proxy', r'the one an honest rule picks $76\%$',
        row['proxy_picked']['realization_rho_mean'], '{:.0f}')
    chk('realization: best proxy', r'proxies of Section~\ref{sec:scope} reaches $83\%$',
        row['proxy_best']['realization_rho_mean'], '{:.0f}')
    chk('realization: transferred', r'another dataset $92\%$',
        row['transfer']['realization_rho_mean'], '{:.0f}')
    chk('realization: ridge recalibrated', r'recalibrated target labels $98\%$',
        row['ridge_recal_1000']['realization_rho_mean'], '{:.0f}')

    # --- the trained predictors, refitted under the same draws by s68_learned_predictors.py
    lp = J('learned_predictors')
    _lit = (r'gradient-boosted trees $99\%$, a path-encoded neural predictor $96\%$ and a '
            r'random forest $94\%$')
    for key, lab in (('learned_gbdt', 'boosted trees'), ('learned_mlp_path', 'neural'),
                     ('learned_rf', 'forest')):
        chk('realization: %s' % lab, _lit, row[key]['realization_rho_mean'], '{:.0f}')
    #: the body states the margin boosting has over the ridge on each dataset, so each one is
    #: checked rather than only the claim that a margin exists
    _gap = r'by $0.011$,\n$0.013$ and $0.015$ in $\rho$'.replace('\\n', ' ')
    for ds in NB:
        g = (lp['results'][ds]['1000']['gbdt']['recal']['rho']
             - lp['results'][ds]['1000']['ridge2']['recal']['rho'])
        chk('boosting over the ridge (%s)' % ds, _gap, g, '{:.3f}')
        if g <= 0:
            BAD.append('the body says gradient boosting passes the ridge on all three '
                       'datasets at 1,000 labels; on %s it does not' % ds)
    _gain = (r'$+0.169$ in $\rho$ on average to the neural predictor, $+0.065$ to the boosted '
             r'trees and $+0.122$ to the ridge')
    for key, lab in (('mlp_path', 'neural predictor'), ('gbdt', 'boosted trees'),
                     ('ridge2', 'ridge')):
        chk('recalibration gain (%s)' % lab, _gain,
            lp['findings']['recalibration_gain_rho'][key]['mean'], '{:.3f}')
    #: the body's two budget-independent claims, each a flag the result file computes
    if not lp['checks']['nothing_fitted_beats_the_exact_reference']:
        BAD.append('the body says no fitted arm passes the exact degree-2 projection in any '
                   'cell; learned_predictors.json says one does')
    if not lp['checks']['recalibration_helps_every_arm_on_average']:
        BAD.append('the body says the recalibration is worth as much to the trained '
                   'predictors as to the ridge; learned_predictors.json says an arm loses '
                   'from it on average')
    if not lp['checks']['ridge_matches_s53']:
        BAD.append('the ridge row of s68_learned_predictors.py no longer equals the one '
                   's53_spectrum.py publishes, so the two are not running one protocol and '
                   'the comparison in Table IV is not a head-to-head (worst gap %.4f)'
                   % lp['checks']['ridge_vs_s53_worst_abs_rho_gap'])
    #: the supplement states the hundred-label numbers it draws the ordering from
    _s100 = ('$0.795$, $0.782$ and\n$0.778$ in $\\rho$ against gradient '
             "boosting's $0.636$, $0.630$ and $0.617$").replace('\n', ' ')
    for ds in NB:
        chk('ridge at 100 (%s)' % ds, _s100,
            lp['results'][ds]['100']['ridge2']['recal']['rho'], '{:.3f}')
        chk('boosting at 100 (%s)' % ds, _s100,
            lp['results'][ds]['100']['gbdt']['recal']['rho'], '{:.3f}')
    _m100 = r'reaches $0.834$, $0.814$ and $0.797$ there'
    for ds in NB:
        chk('neural predictor at 100 (%s)' % ds, _m100,
            lp['results'][ds]['100']['mlp_path']['recal']['rho'], '{:.3f}')
    chk('realization: abstract', r'reaches $98\%$ of the rank fidelity of the exact degree-2',
        row['ridge_recal_1000']['realization_rho_mean'], '{:.0f}')
    chk('abstract: boosting', r'gradient-boosted trees given the same labels reach $99\%$',
        row['learned_gbdt']['realization_rho_mean'], '{:.0f}')
    if re_['checks']['any_ranker_beats_the_exhibited_optimum']:
        BAD.append('realization: a ranker passed the exhibited degree-2 optimum, but the '
                   'paper says none does')
    else:
        OK.append('realization: no ranker passes the exhibited optimum, as the paper says')

    # ---- what determining a degree-2 model costs
    chk('dimension: nb201', r'$265$ free parameters against $15{,}625$',
        dim['rows']['NAS-Bench-201']['dim_degree2'], '{:d}')
    chk('dimension: nb201 contributions', r'$265$ parameters here, not $15{,}625$',
        dim['rows']['NAS-Bench-201']['dim_degree2'], '{:d}')
    chk('dimension: fractalnet', r'the FractalNet space $942$ against',
        dim['rows']['FractalNet record']['dim_degree2'], '{:d}')
    chk('dimension: pair cells', r'$1{,}514$ pair cells',
        dim['rows']['FractalNet record']['pair_cells'],
        lambda v: '%d{,}%03d' % (v // 1000, v % 1000))
    for ds, lab in (('cifar10', 'c10'), ('cifar100', 'c100')):
        chk('dimension: labels per pair cell (%s)' % lab,
            r'at $3.5$ and $3.1$ labels', dim['coverage'][ds]['labels_per_cell'], '{:.1f}')
        chk('dimension: grid fraction (%s)' % lab,
            r'from the $0.010\%$ and $0.009\%$ of the grid they touched',
            100 * dim['coverage'][ds]['grid_fraction'], '{:.3f}')
    if not dim['checks_coverage']['every_pair_cell_covered']:
        BAD.append('dimension: the paper claims every pair cell was covered, and it was not')
    else:
        OK.append('dimension: every pair cell covered, as the paper says')

    # ---- what the single-column search protocol costs
    pb = sr['protocol_bias']
    lo = min(pb['optimism'][a_]['100'] for a_ in sr['algorithms'])
    hi = max(pb['optimism'][a_]['100'] for a_ in sr['algorithms'])
    chk('protocol bias: low', r'inflates every algorithm by\n$0.13$ to $0.33$ points'
        .replace('\\n', ' '), lo, '{:.2f}')
    chk('protocol bias: high', r'$0.13$ to $0.33$ points', hi, '{:.2f}')
    chk('protocol bias: spread', r'a spread of $0.21$ across the four',
        pb['spread_across_algorithms']['100'], '{:.2f}')
    if not pb['every_swap_is_inside_an_overlap']:
        BAD.append('protocol bias: the paper says every reordering is inside an interval '
                   'overlap, and one is not')
    else:
        OK.append('protocol bias: every reordering is inside an interval overlap')
    n_sig = len(sr['checks']['recipe_beats_random_significantly_at'])
    n_all = len(sr['datasets']) * len(sr['protocol']['budgets'])
    chk('search: cells where the recipe leads random',
        r'the procedure is ahead in $21$ of $21$', n_sig, '{:d}')
    if n_sig != n_all:
        BAD.append('search: %d of %d cells are significant, so "every one" is wrong'
                   % (n_sig, n_all))
    #: the interval bounds, not only the point estimate. The bounds come from a bootstrap
    #: stream, so they move whenever the contrasts are recomputed in a different order, and
    #: a quoted bound that nothing checks is the next thing to go stale.
    def bounds(c):
        return '[%.2f,%.2f]' % (c['ci'][0], c['ci'][1])

    def abs_bounds(c):
        lo, hi = sorted(abs(x) for x in c['ci'])
        return '[%.2f,%.2f]' % (lo, hi)

    for ds, lab, base, fmt in (
            ('cifar10', 'c10', 'vs_random', bounds),
            ('cifar100', 'c100', 'vs_random', bounds),
            ('ImageNet16-120', 'in16', 'vs_random', bounds),
            ('cifar10', 'c10', 'vs_evolution', bounds),
            ('ImageNet16-120', 'in16', 'vs_evolution', bounds),
            ('cifar100', 'c100', 'vs_evolution', abs_bounds)):
        c = sr['results'][ds]['recipe']['100'][base]
        chk('search interval: %s %s' % (lab, base), fmt(c))
    #: local search is now carried by the ledger rather than by three numbers at one budget,
    #: and the ledger is checked above; what remains quoted at N=100 is the contrast against
    #: regularized evolution, interval and all
    chk('search: recipe vs evolution c10', r'$+0.02$ $[-0.02,0.06]$ on CIFAR-10',
        sr['results']['cifar10']['recipe']['100']['vs_evolution']['diff'], '{:+.2f}')
    chk('search: recipe vs evolution in16', r'$+0.05$ $[-0.01,0.11]$ on ImageNet16-120',
        sr['results']['ImageNet16-120']['recipe']['100']['vs_evolution']['diff'], '{:+.2f}')
    chk('search: recipe vs evolution c100', r'behind on CIFAR-100 by $0.14$ $[0.06,0.22]$',
        abs(sr['results']['cifar100']['recipe']['100']['vs_evolution']['diff']), '{:.2f}')

    # ---- the two counts of the released learning curves, which are different quantities
    #: 8,654 architecture-RUN pairs in the released file; 8,652 DISTINCT architectures, two
    #: CIFAR-10 architectures having been evaluated in two runs each. The budget policies draw
    #: from the distinct set. Both numbers are quoted in the supplement and they must not be
    #: allowed to converge on each other, which is what these two checks are for.
    cc = J('halving')['curve_counts']

    def comma(v):
        return '%d{,}%03d' % (v // 1000, v % 1000)

    chk('learning curves: rows', r'\texttt{predicted\_fitness}. 30{,}246 rows',
        cc['rows'], comma)
    chk('learning curves: architecture-run pairs',
        r'cover 10{,}082 architecture-run pairs', cc['architecture_run_pairs'], comma)
    chk('learning curves: distinct architectures',
        r'pairs are $8{,}427$ \emph{distinct} architectures',
        cc['distinct_architectures'], comma)
    if cc['architecture_run_pairs'] == cc['distinct_architectures']:
        BAD.append('learning curves: the two counts are equal, so the sentence that '
                   'distinguishes them is wrong')

    # ---- which monotone map
    ls = mo['landscapes']
    chk('monotone: cost of the wrong criterion',
        r'and $0.03$ accuracy points at a shortlist of ten on NAS-Bench-Macro',
        mo['checks']['max_regret_cost_of_choosing_on_the_ceiling'], '{:.2f}')
    chk('monotone: cost of not correcting', r'staying on raw accuracy costs up to $3.18$',
        mo['checks']['max_cost_of_not_recalibrating'], '{:.2f}')
    chk('monotone: rank vs normal scores', r'rank fidelities within $0.004$',
        max(abs(ls[d]['summary']['ranks_minus_normal_scores_rho']) for d in ls),
        lambda v: '%.3f' % (round(v, 3) + (0.001 if round(v, 3) < v else 0)))
    chk('monotone: ceilings still differ', r'ceilings differ by up to $1.4$ points'
        .replace('\\n', ' '),
        mo['checks']['max_rank_vs_normal_scores_a_le_2_gap'], '{:.1f}')
    for flag, msg in (
            ('the_two_criteria_name_different_maps_everywhere',
             'the paper says the two criteria name different maps on every landscape'),
            ('every_landscape_prefers_a_rank_based_map',
             'the paper says every landscape prefers a rank-based map'),
            ('rank_based_maps_decide_alike_everywhere',
             'the paper says ranks and normal scores decide alike')):
        if not mo['checks'][flag]:
            BAD.append('monotone: %s, and monotone.json says otherwise' % msg)
        else:
            OK.append('monotone: %s, confirmed' % msg)

    #: the body now says the criteria part on NAS-Bench-Macro and nowhere else, which is a
    #: statement about the decision rather than about which map wins a name
    free = set(mo['checks']['landscapes_where_choosing_on_the_ceiling_costs_nothing'])
    costly = set(mo['checks']['landscapes_where_choosing_on_the_ceiling_costs_something'])
    if free == {'cifar10', 'cifar100', 'ImageNet16-120'} and costly == {'nasbench_macro'}:
        OK.append('monotone: choosing on the ceiling is free on the three NAS-Bench-201 '
                  'tasks and costly only on NAS-Bench-Macro, as the body says')
    else:
        BAD.append('monotone: the body says choosing on the ceiling is free on the three '
                   'NAS-Bench-201 tasks and costly only on NAS-Bench-Macro; monotone.json '
                   'says free on %s and costly on %s' % (sorted(free), sorted(costly)))


def main():
    hl = J('halving')
    sur, sig, tr = J('surrogate'), J('signal_anatomy'), J('transfer')
    si, rc, wm = J('sitl'), J('recipe'), J('warmstart')
    hal = J('halving')
    rg, nb = J('regret'), J('nb201')

    # ---------------- Section IV: the ranking signal, now in regret
    chk('surrogate rho c10 (half-split)', r'the genotype surrogate reaches $\rho=0.967$ and $0.936$',
        rg[A]['rho']['surrogate'])
    chk('surrogate rho c100 (half-split)', r'reaches $\rho=0.967$ and $0.936$',
        rg[B]['rho']['surrogate'])
    chk('epoch1 rho both', r'the $0.937$ and $0.890$ of a full training epoch',
        rg[B]['rho']['epoch1'], lambda v: '%.3f' % v)
    chk('predictor rho c10', r'$2.8$ times the $0.578$ and $0.334$ of the LLM predictor',
        rg[A]['rho']['predictor'])
    chk('surrogate 5-fold', r'it reaches $0.972$ and $0.945$',
        sur[A]['rho']['surrogate'])
    chk('rho10 surrogate', r'the surrogate falls to $\rho_{10}=0.620$ and $0.749$',
        rg[A]['rho_top10']['surrogate'])
    chk('rho10 epoch', r'a measured epoch to $0.560$ and $0.690$',
        rg[A]['rho_top10']['epoch1'])
    chk('regret top10', r'top-ten shortlist gives up $0.07$ and $0.00$ accuracy points',
        rg[A]['regret']['surrogate']['10']['mean'], '{:.2f}')
    chk('regret top20', r'its top twenty $0.05$ and $0.00$',
        rg[B]['regret']['surrogate']['20']['mean'], '{:.2f}')
    chk('kstar', r'$k^{\star}=10$ for the free surrogate on both datasets',
        rg[A]['k_star']['surrogate'], '{:d}')
    chk('kstar fused', r'$5$ and $10$ once fused with a measured epoch',
        rg[A]['k_star']['fused'], '{:d}')
    chk('loo cross-run', r'$\rho=0.966\pm0.007$ and $0.939\pm0.015$',
        sur[A]['loo']['mean'])
    chk('identifiability cells', r'$1{,}514$ cells on FractalNet',
        sig[A]['identifiability']['pair_cells'],
        lambda v: '%d{,}%03d' % (v // 1000, v % 1000) if v >= 1000 else '%d' % v)
    chk('grid fraction', r"dataset's runs touched $0.010\%$ and $0.009\%$ of the grid",
        sig[A]['identifiability']['grid_fraction'] * 100, '{:.3f}')
    chk('grid fraction c100', r'$0.010\%$ and $0.009\%$',
        sig[B]['identifiability']['grid_fraction'] * 100, '{:.3f}')
    chk('labels per cell', r'at $3.5$ and $3.1$ labels per cell',
        sig[A]['identifiability']['labels_per_cell'], '{:.1f}')
    chk('additive r2', r'reaches $R^2=0.757$ and $0.669$',
        sig[A]['identifiability']['additive']['r2'])
    chk('pairwise r2', r'the interactions take it to $0.934$ and $0.891$',
        sig[A]['identifiability']['pairwise']['r2'])
    chk('sample eff 250', r'250 architectures already give $\rho=0.889$ and $0.831$',
        sur[A]['sample_efficiency']['250'][0])
    chk('sample eff 1000', r'$0.946$ and $0.909$ at 1{,}000',
        sur[A]['sample_efficiency']['1000'][0])
    chk('residual epoch', r'($+0.489$ and $+0.622$)', sig[A]['residual']['epoch1_vs_truth'])
    chk('residual predictor', r'almost nothing ($+0.164$ and $+0.102$)',
        sig[A]['residual']['predictor_vs_truth'])
    chks('fusion rho (second mention)', r'at $\rho=0.976$ and $0.958$ with $k^{\star}=5$ and $5$',
         (rg[A]['rho']['fused'], rg[B]['rho']['fused'], rg[A]['k_star']['fused'], rg[B]['k_star']['fused']),
         lambda v: '$\\rho=%.3f$ and $%.3f$ with $k^{\\star}=%d$ and $%d$' % v)
    chk('fusion rho', r'($\rho=0.976$ and $0.958$', rg[A]['rho']['fused'])
    chk('fusion rho10', r'0.656', rg[A]['rho_top10']['fused'])

    # ---------------- Section V: transfer, own space
    # Section V quotes every pair target-CIFAR-10 first, which is the order the rest of the
    # paper uses; transfer.json is keyed source->target, so C100->C10 is the CIFAR-10 value.
    chk('transfer c100->c10', r'the surrogate ranks them at $\rho=0.891$',
        tr['dataset']['cifar100->cifar10']['few_shot']['0']['transfer'])
    chk('transfer c10->c100', r'$\rho=0.891$ and $0.873$',
        tr['dataset']['cifar10->cifar100']['few_shot']['0']['transfer'])
    chk('epoch on target c10', r'a full epoch on every candidate gives $0.934$ and $0.889$',
        tr['dataset']['cifar100->cifar10']['epoch1_on_target'])
    chk('epoch on target c100', r'gives $0.934$ and $0.889$',
        tr['dataset']['cifar10->cifar100']['epoch1_on_target'])
    chk('scratch 500 c10', r'on 500 target labels gives $0.915$\nand $0.872$'
        .replace('\\n', ' '),
        tr['dataset']['cifar100->cifar10']['few_shot']['500']['scratch'])
    chk('scratch 500 c100', r'gives $0.915$ and $0.872$',
        tr['dataset']['cifar10->cifar100']['few_shot']['500']['scratch'])
    chk('few shot 25 c10', r'take it to $0.896$ and $0.880$',
        tr['dataset']['cifar100->cifar10']['few_shot']['25']['transfer'])
    chk('few shot 25 c100', r'$0.896$ and $0.880$',
        tr['dataset']['cifar10->cifar100']['few_shot']['25']['transfer'])
    chk('few shot 500 c10', r'five hundred to $0.946$ and $0.924$',
        tr['dataset']['cifar100->cifar10']['few_shot']['500']['transfer'])
    chk('few shot 500 c100', r'to $0.946$ and $0.924$',
        tr['dataset']['cifar10->cifar100']['few_shot']['500']['transfer'])
    chk('arm transfer', r'it reaches $0.930$ and $0.849$',
        tr['arm'][A]['baseline->llm']['rho'])
    chk('epoch1 labels', r'ranks the 3-epoch reference at $0.934$ and $0.863$',
        tr['reference'][A]['fit_on_epoch1_rank_epoch3'])
    # the section's convention is (target CIFAR-10, target CIFAR-100), so the values are
    # transfer.json's cifar100->cifar10 and cifar10->cifar100 in that order
    chk('top50 recovery c10', r'recovers $64\%$ and $62\%$ of the target',
        tr['dataset']['cifar100->cifar10']['top50'], lambda v: '%d\\%%' % round(100 * v))
    chk('top50 recovery c100', r'recovers $64\%$ and $62\%$ of the target',
        tr['dataset']['cifar10->cifar100']['top50'], lambda v: '%d\\%%' % round(100 * v))

    # ---------------- Section V: NAS-Bench-201
    chk('nb201 1000', r'the model reaches $\rho=0.737$, $0.856$ and $0.910$ from a thousand '
        r'labels', nb['fidelity'][NB[0]]['1000']['rho'])
    chk('nb201 1000 c100', r'$0.856$ and $0.910$ from a thousand labels',
        nb['fidelity'][NB[1]]['1000']['rho'])
    chk('nb201 1000 in16', r'$0.910$ from a thousand labels',
        nb['fidelity'][NB[2]]['1000']['rho'])
    chk('nb201 100', r'$0.615$, $0.713$ and $0.766$ from a hundred',
        nb['fidelity'][NB[0]]['100']['rho'])
    chk('nb201 100 c100', r'$0.713$ and $0.766$ from a hundred',
        nb['fidelity'][NB[1]]['100']['rho'])
    chk('nb201 params proxy', r"0.750",
        nb['fidelity'][NB[0]]['params']['rho'])
    chk('nb201 params c100', r'0.725',
        nb['fidelity'][NB[1]]['params']['rho'])
    chk('nb201 held out', r'twenty architectures drawn at \emph{random} are already within $1.24$')
    chk('nb201 seeds', r'20 random splits, as everywhere below', nb['n_seeds'],
        lambda v: '%d random splits' % v)
    chk('nb201 direction down', r'averages $\rho=0.890$', nb['direction']['down'])
    chk('nb201 direction up', r'against $0.745$ out of CIFAR-10', nb['direction']['up'])
    chk('nb201 in16->c10', r'ranks CIFAR-10 at $0.887$ against $0.737$',
        nb['transfer']['ImageNet16-120->cifar10']['rho'])
    chk('nb201 in16->c10 own', r'at $0.887$ against $0.737$',
        nb['transfer']['ImageNet16-120->cifar10']['rho_fitted_here'])
    chk('nb201 in16->c100', r'CIFAR-100 at $0.892$ against $0.856$',
        nb['transfer']['ImageNet16-120->cifar100']['rho'])
    chk('nb201 c100->c10', r'CIFAR-100 ranks CIFAR-10 at $0.843$ against $0.737$',
        nb['transfer']['cifar100->cifar10']['rho'])
    chk('nb201 downhill count', r'All three transfers from a harder dataset to an '
        r'easier one beat the same', nb['direction']['beats_own_down'],
        lambda v: 'All three' if v == 3 else 'All %d' % v)
    chk('nb201 random rg20', r'already within $1.24$, $2.98$ and $2.92$ points',
        nb['regret_random'][NB[0]]['20']['mean'], '{:.2f}')
    chk('nb201 random rg20 c100', r'$2.98$ and $2.92$ points',
        nb['regret_random'][NB[1]]['20']['mean'], '{:.2f}')
    chk('own random rg20', r'against $5.31$ and $14.31$ in our own space',
        rg[A]['regret']['random']['20']['mean'], '{:.2f}')
    chk('nb201 own rg20 c10', r'shortlists CIFAR-10 at $1.15$ points against a random $1.24$',
        nb['regret'][NB[0]]['20']['mean'], '{:.2f}')
    chk('nb201 transfer rg20', r'reaches $0.14$ and $0.09$ points on CIFAR-10 and CIFAR-100',
        nb['transfer']['ImageNet16-120->cifar10']['regret20'], '{:.2f}')
    chk('nb201 transfer rg20 c100', r'$0.09$ points on CIFAR-10 and CIFAR-100',
        nb['transfer']['ImageNet16-120->cifar100']['regret20'], '{:.2f}')
    chk('nb201 features', r'405 features', nb['n_features'], lambda v: '%d' % v)
    chk('nb201 n', r'$15{,}625$ architectures', nb['n_arch'],
        lambda v: '15{,}625' if v == 15625 else '%d' % v)

    # ---------------- Section VI: the budget
    chk('recipe as run', r'returns $80.67$ and $41.48$',
        rc[A]['cells']['A_train100']['acc'], '{:.2f}')
    # the stage accuracies rather than the increments: rounding each increment to two
    # decimals makes them sum to 4.29 on CIFAR-10 while the total is 4.28, so the text
    # quotes the cumulative value at each stage and the total once.
    chk('recipe screen c10', r'candidates raises it to $82.71$ and $45.81$',
        rc[A]['cells']['B_screen800']['acc'], '{:.2f}')
    chk('recipe screen c100', r'candidates raises it to $82.71$ and $45.81$',
        rc[B]['cells']['B_screen800']['acc'], '{:.2f}')
    chk('recipe halving c10', r'raises it to $83.41$ and $47.61$',
        rc[A]['cells']['C_screen_halve']['acc'], '{:.2f}')
    chk('recipe halving c100', r'raises it to $83.41$ and $47.61$',
        rc[B]['cells']['C_screen_halve']['acc'], '{:.2f}')
    chk('recipe total', r'the full recipe returns $+2.74$ and $+6.13$ points',
        rc[A]['cells']['C_screen_halve']['acc'] - rc[A]['cells']['A_train100']['acc'],
        '{:.2f}')
    chk('recipe total c100', r'$+2.74$ and $+6.13$ points',
        rc[B]['cells']['C_screen_halve']['acc'] - rc[B]['cells']['A_train100']['acc'],
        '{:.2f}')
    chk('recipe fused null', r'rule of the halving stage is worth $0.00$ and\n$-0.03$ points'
        .replace('\\n', ' '),
        rc[A]['cells']['D_screen_halve_fused']['acc']
        - rc[A]['cells']['C_screen_halve']['acc'], lambda v: ('%.2f' % v).replace('-0.00', '0.00'))
    chk('recipe full fidelity', r'by $0.67$ and $2.55$ points',
        rc[A]['cells']['A_train100']['acc'] - rc[A]['cells']['ref_three_epochs_33']['acc'],
        '{:.2f}')
    chk('recipe no training', r'returns $+1.62$ and $+3.57$',
        rc[A]['cells']['ref_surrogate_only']['acc']
        - rc[A]['cells']['A_train100']['acc'], '{:.2f}')
    chk('halving curves', r'comes within $0.00$ and $0.02$ points of the 3-epoch reference '
        r'for $46\%$ of its cost', hl[A]['cells']['halving_25']['regret'], '{:.2f}')
    chk('halving flat', r'against $67\%$ and $0.35$ and $0.42$ points for two flat epochs',
        hl[A]['cells']['epoch2']['regret'], '{:.2f}')
    chk('halving 50', r'a top-$50\%$ promotion closes to within $0.000$ points of the '
        r'3-epoch reference at $58\%$ of its cost',
        hl[A]['cells']['halving_50']['regret'], '{:.3f}')
    chks('predictor cost', r'is worth $18.06$ and $27.02$ points',
         J('extras')['top1_epoch_vs_predictor'][A]['mean'], '{:.2f}')
    chks('predictor cost c100', r'$18.06$ and $27.02$ points',
         J('extras')['top1_epoch_vs_predictor'][B]['mean'], '{:.2f}')
    chk('learning curves n', r'$8{,}427$ distinct architectures of Section')

    # ---------------- Section VII: model in the loop
    chk('sitl prior rho', r'starts at $\rho=0.861$', si[A]['prior_rho'])
    chk('sitl gain 120', r'gains $\mathbf{+4.62}$ and $\mathbf{+4.42}$ points',
        si[A]['120']['transfer'] - si[A]['120']['plain'], '{:.2f}')
    chk('sitl gain 260', r'$+0.82$ and $+1.13$ at 260,',
        si[A]['260']['transfer'] - si[A]['260']['plain'], '{:.2f}')
    chk('sitl gain 520', r'$-0.79$ and $+0.23$ at 520,',
        si[A]['520']['transfer'] - si[A]['520']['plain'], '{:.2f}')
    chk('sitl scratch fid', r'stalls at $\rho=0.59$ and $0.47$',
        si[A]['120']['scratch_fid'], '{:.2f}')
    chk('sitl scratch loss', r'by $6.64$ and $7.68$ points at 120 architectures',
        -si[A]['120']['scratch_delta'], '{:.2f}')
    # the same quantity at the largest budget: the main paper quotes both ends, because
    # "7 to 14" in the contribution list has to be a range the table actually spans
    chk('sitl scratch loss 520', r'widening to $12.44$ and $14.30$ at 520',
        -si[A]['520']['scratch_delta'], '{:.2f}')
    chk('sitl scratch loss 520 c100', r'widening to $12.44$ and $14.30$ at 520',
        -si[B]['520']['scratch_delta'], '{:.2f}')
    chk('sitl explore fid', r'raises the model to $0.64$ and $0.55$',
        si[A]['120']['explore_fid'], '{:.2f}')
    PL = {d: wm[d]['predictor_level'] for d in (A, B)}       # the predictor's own fidelity
    chk('warmstart level', r"own predictor ($\rho=0.58$ and $0.33$)", (PL[A], PL[B]),
        lambda v: '$\\rho=%s$ and $%s$' % v)
    chk('warmstart loss', r'loses $4.11$ and $5.91$ points',
        (-wm[A]['asrun'][PL[A]]['loss'], -wm[B]['asrun'][PL[B]]['loss']),
        lambda v: '$%.2f$ and $%.2f$' % v)
    chk('warmstart warm', r'($-0.61$ and $+2.11$ against an exact signal)',
        (wm[A]['asrun'][PL[A]]['loss_warm'], wm[B]['asrun'][PL[B]]['loss_warm']),
        lambda v: '$%+.2f$ and $%+.2f$' % v)
    chk('warmstart worst', r'loses at most $0.61$ points',
        -min(wm[d]['asrun'][PL[d]]['loss_warm'] for d in (A, B)), '{:.2f}')
    chk('warmstart recovered', r'recovering $85\%$ and $136\%$',
        (100 * wm[A]['asrun'][PL[A]]['recovered'], 100 * wm[B]['asrun'][PL[B]]['recovered']),
        lambda v: '$%.0f\\%%$ and $%.0f\\%%$' % v)
    chks('prereg disclosure recipe', r'two quantities are $+2.74$ and $+6.13$ points',
         (rc[A]['cells']['C_screen_halve']['acc'] - rc[A]['cells']['A_train100']['acc'],
          rc[B]['cells']['C_screen_halve']['acc'] - rc[B]['cells']['A_train100']['acc']),
         lambda v: '$%+.2f$ and $%+.2f$ points' % v)
    chks('prereg disclosure loop', r'$+4.62$ and $+4.42$ for the seeded loop',
         (si[A]['120']['transfer'] - si[A]['120']['plain'], si[B]['120']['transfer'] - si[B]['120']['plain']),
         lambda v: '$%+.2f$ and $%+.2f$' % v)
    chk('emulation warning', r'predicts a gain for a policy that in fact loses $7$ to $14$ points',
        (-si[A]['120']['scratch_delta'], -si[B]['520']['scratch_delta']),
        lambda v: 'loses $%d$ to $%d$ points' % (round(v[0]), round(v[1])))

    # ---------------- Section VIII: the benchmark

    # ---------------- Section X: the space, and past its edge

    # ---------------- Section IX: the acceptance gate

    # ---------------- method and protocol, quoted consistently
    chk('n arch', r'10{,}082\nper-architecture records'.replace('\\n', ' '))
    chk('n features', r'$1{,}572$ features in total on FractalNet')
    chk('grid size', r'$4.2\times10^{7}$ points in all')
    chk('n runs', r'34 runs in total')
    chk('fitness provenance', r'all $10{,}082$ evaluations, whose $10{,}086$ fitness lines')
    chk('splits', r'40 random half-splits')
    chk('censoring', r'$6.8$--$16.0\%$ of attempts on CIFAR-10, $20.7$--$25.5\%$')

    # the two extrapolators must agree in sign on every forecast, and the paper quotes how
    # far apart they are on the two positive ones

    # ---------------- the supplement: the numbers the ten-page limit pushed out of main.tex
    st = J('stats')
    chks('supp predictor ge90', r'mean 3-epoch accuracy of 16.70\%',
         J('extras')['raw_high_band_cifar100']['mean_high'], '{:.2f}')
    chks('supp population mean', r'against a population mean of 15.42\%',
         J('extras')['raw_high_band_cifar100']['mean_all'], '{:.2f}')
    chks('supp random forest', r'reaches $\rho=0.955$ and $0.925$',
         sur[A]['rho']['random_forest'], '{:.3f}')
    chks('supp random forest c100', r'reaches $\rho=0.955$ and $0.925$',
         sur[B]['rho']['random_forest'], '{:.3f}')
    chks('supp ridge cv', r'below the ridge ($0.972$ and $0.945$)',
         sur[A]['rho']['surrogate'], '{:.3f}')
    chks('supp epoch1 ref', r'above the one-epoch proxy ($0.937$ and $0.890$)',
         sur[A]['rho']['epoch1'], '{:.3f}')
    chks('supp replay timeout', r'a per-call time-out of 5\,ms')
    #: the halving table's own caption, against the curve counts s17_halving.py used
    chks('supp curves', r'the recovered learning curves of the 8{,}427 distinct',
         hl[A]['n_curves'] + hl[B]['n_curves'], lambda v: '%d{,}%03d' % (v // 1000, v % 1000))
    rcj = J('record_check')
    chks('record rows', r'has one row for each of the $10{,}082$ architectures the runs evaluated',
         rcj['rows'], lambda v: '$%d{,}%03d$' % (v // 1000, v % 1000))
    chks('record frozen check', r'for the $2{,}988$ frozen-control rows, genes identical',
         rcj['frozen_rows_checked_against_logged_chromosome'], lambda v: '$%d{,}%03d$' % (v // 1000, v % 1000))
    # ---- the robustness checks of Supplementary Section S18 and the transfer arms
    rb, sp = J('robustness'), J('spectrum')
    rc10 = sp['recalibration']['cifar10']['budgets']['1000']
    chk('transfer recal closes the gap', r'from $0.879$ to $0.908$',
        (rc10['transfer_raw']['rho'], rc10['transfer_recal']['rho']),
        lambda v: 'from $%.3f$ to $%.3f$' % v)
    r16 = sp['recalibration']['ImageNet16-120']['budgets']['500']
    chks('IN16 transfer recal regret', r'($0.26$ against $0.60$ points at $500$',
         (r16['transfer_recal']['regret10'], r16['own_recal']['regret10']),
         lambda v: '($%.2f$ against $%.2f$' % v)
    eq = rb['equivalence']
    chks('equivalence classes', r'fall into $8{,}765$ classes', eq['n_classes'],
         lambda v: '$%d{,}%03d$' % (v // 1000, v % 1000))
    chks('class-mean spectrum', r'to $67.9$, $78.2$ and $88.5\%$',
         tuple(100 * eq['datasets'][d]['class_mean_acc']['le2'] for d in NB),
         lambda v: '$%.1f$, $%.1f$ and $%.1f\\%%$' % v)
    moves = [abs(eq['datasets'][d][a]['le2'] - eq['datasets'][d][b]['le2']) * 100
             for d in NB for a, b in (('raw_acc', 'class_mean_acc'), ('raw_recal', 'class_mean_recal'))]
    chk('equivalence max move (main)', r'changes no share by more than $0.7$ points',
        max(moves), lambda v: '$%.1f$' % (round(v, 1) + (0.1 if round(v, 1) < v else 0)))
    chks('equivalence max move (supp)', r'the recalibrated shares by at most $0.7$ points',
         max(moves), lambda v: '$%.1f$' % (round(v, 1) + (0.1 if round(v, 1) < v else 0)))
    chks('equivalence within-class share', r'carry $0.03$, $0.12$ and $0.29\%$',
         tuple(100 * eq['datasets'][d]['within_class_variance_share'] for d in NB),
         lambda v: '$%.2f$, $%.2f$ and $%.2f\\%%$' % v)
    rn = rb['rank_null']
    chks('rank null keeps apart', r'Ranked, they keep $0.87$ and $0.29$',
         (rn['low_order']['ranked'], rn['high_order']['ranked']), lambda v: '$%.2f$ and $%.2f$' % v)
    chks('rank null compression', r'down to $0.32$', rn['low_order']['compressed_2'], '{:.2f}')
    st = rb['scope_test']
    lo = min(st[d]['10']['share_not_positive'] for d in NB) * 100
    chk('scope test (main)', r'non-positive in $74.5$--$99.5\%$ of draws',
        (lo, max(st[d]['5']['share_not_positive'] for d in NB) * 100),
        lambda v: '$%.1f$--$%.1f\\%%$' % v)
    chks('scope test at 25 (supp)', r'$17.5$--$19.0\%$ at $25$',
         (min(st[d]['25']['share_not_positive'] for d in NB) * 100,
          max(st[d]['25']['share_not_positive'] for d in NB) * 100),
         lambda v: '$%.1f$--$%.1f\\%%$' % v)
    chks('scope test at 50 (supp)', r'at most $0.5\%$ at $50$',
         max(st[d]['50']['share_not_positive'] for d in NB) * 100, '{:.1f}')
    if any(st[d][n]['share_not_positive'] > 0 for d in NB for n in ('100', '200')):
        BAD.append('scope test: R2 is non-positive in some draw from 100 labels, but the text says none')
    else:
        OK.append('scope test: never non-positive from 100 labels')
    # ---- shortlist-size sensitivity (s75), proxy cost, and the 26-fold contrast
    ks = J('k_sensitivity')
    chk('k: reversal counts (main)', r'$10$, $6$ and $4$ of the eighteen at $k=5$, $20$ and $50$',
        tuple(ks['ladder'][k]['pairs_lower_ceiling_lower_regret'] for k in ('5', '20', '50')),
        lambda v: '$%d$, $%d$ and $%d$ of the eighteen' % v)
    chks('k: reversal counts (supp)', r'in $10$, $9$, $6$ and $4$ of its eighteen',
         tuple(ks['ladder'][k]['pairs_lower_ceiling_lower_regret'] for k in ('5', '10', '20', '50')),
         lambda v: '$%d$, $%d$, $%d$ and $%d$' % v)
    if any(ks['transfer'][k]['own_beats_transfer_raw'] != ks['transfer'][k]['cells'] for k in ks['transfer']):
        BAD.append('k: own recalibrated labels do not beat raw-scale transfer in every cell at every k')
    else:
        OK.append('k: own recalibrated labels beat raw-scale transfer in every cell at every k')
    chks('k: against recalibrated transfer', r'less than the recalibrated transfer in $9$, $8$ and $8$',
         tuple(ks['transfer'][k]['own_beats_transfer_recal'] for k in ('5', '20', '50')),
         lambda v: '$%d$, $%d$ and $%d$' % v)
    chks('k: g2 recal at k=5', r'and at most $0.46$ at $k=5$',
         max(ks['ladder']['5']['g2_regret_recal'].values()), '{:.2f}')
    zc = J('zc_nb201')
    pick = zc['proxy_selection']['pick_source']      # the honest rule's choice per dataset
    chk('proxy scoring cost', r'costs $0.74$, $0.76$ and $3.58$ GPU-hours',
        tuple(zc['cost'][d][pick[d]]['hours_pool_14625'] for d in NB),
        lambda v: '$%.2f$, $%.2f$ and $%.2f$' % v)
    c1k = J('spectrum')['recalibration']['cifar10']['budgets']['1000']
    chk('twenty-six-fold', r'twenty-six-fold reduction in regret (from $1.32$ to $0.05$ points)',
        (c1k['own_raw']['regret10'], c1k['own_recal']['regret10']),
        lambda v: ('twenty-six-fold' if round(v[0] / v[1]) == 26 else '%d-fold' % round(v[0] / v[1]))
        + ' reduction in regret (from $%.2f$ to $%.2f$ points)' % v)
    # ---- the third complete factorial, NATS-Bench SSS (s63); checked only if it was computed
    ex = os.path.join(common.RES, 'extra_grids.json')
    if os.path.exists(ex) and 'nats_sss' in json.load(open(ex)).get('grids', {}):
        ng = json.load(open(ex))['grids']['nats_sss']
        le2 = tuple(ng['responses'][r]['cumulative'][1] for r in ('acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'))
        le3 = tuple(ng['responses'][r]['cumulative'][2] for r in ('acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'))
        o1 = tuple(ng['responses'][r]['per_order'][0] for r in ('acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'))
        f3 = lambda v: '$%.1f$, $%.1f$ and $%.1f' % v
        chk('NATS-Bench order<=2 (main)', r'puts $96.5$, $99.0$ and $98.1\%$ of the variance', le2, f3)
        chk('NATS-Bench null (main)', r'against a null of $1.6\%$', ng['null']['cumulative'][1], '{:.1f}')
        chk('NATS-Bench in the conclusion', r'at $96.5$ to $99.0\%$', (min(le2), max(le2)),
            lambda v: '$%.1f$ to $%.1f' % v)
        chks('NATS-Bench order<=2 (supp)', r'Order $\le 2$ carries $96.5$, $99.0$ and $98.1\%$', le2, f3)
        chks('NATS-Bench order<=3 (supp)', r'order $\le 3$ $97.3$, $99.2$ and $98.4\%$', le3, f3)
        chks('NATS-Bench nulls (supp)', r'against $1.6$ and $12.1\%$',
             (ng['null']['cumulative'][1], ng['null']['cumulative'][2]), lambda v: '$%.1f$ and $%.1f' % v)
        chks('NATS-Bench order one (supp)', r'($91.9$, $96.2$ and $97.1\%$)', o1, f3)
        chks('NATS-Bench skewness (supp)', r'(skewness $-1.28$, $-1.29$ and $-0.75$)',
             tuple(ng['responses'][r]['compression']['skewness'] for r in ('acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120')),
             lambda v: '$%.2f$, $%.2f$ and $%.2f$' % v)
    else:
        SKIPPED.append('NATS-Bench SSS: results/extra_grids.json has no computed grid')
    zerocost()
    spectrum_claims()
    counts_and_superlatives()
    search_claims()
    novelty_and_decision()
    realization_dimension_and_protocol()
    crossrefs()

    # ---------------- claims the revision retired: they must not reappear
    absent('old framing: space caps accuracy',
           r'the space and not the optimiser is what caps the attainable accuracy')
    absent('old framing: fused promotion null',
           r'Fusing the surrogate into the promotion rule adds a further $0.01$ and $0.02$')
    absent('old claim: GA worse than screening as headline',
           r'The GA as the pipeline ran it reaches $92.53$ and $48.56$ and is therefore '
           r'\emph{worse} than screening alone')
    absent('old claim: predictor verdict', r'the LLM predictor knows almost nothing')
    # The recommendations live in the supplement and the procedure is stated in the
    # Conclusion, so a Recommendations heading must not reappear in main.tex.
    absent('no Recommendations section in the main paper',
           r'\section{Recommendations}')
    chks('the recommendations are in the supplement',
         r'\subsection{The recommendations, in full}')

    print('\n'.join(BAD))
    for note in SKIPPED:
        print('NOTE  %s' % note)
    if MOVED:
        print('\n%d claims are quoted in the supplement rather than in main.tex, which is '
              'where the page limit puts them.' % len(MOVED))
    print('\n%d checks passed, %d moved to the supplement, %d failed%s'
          % (len(OK), len(MOVED), len(BAD),
             ', %d skipped' % len(SKIPPED) if SKIPPED else ''))
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
