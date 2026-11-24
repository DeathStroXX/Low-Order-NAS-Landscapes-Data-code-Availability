"""Cross-reference consistency between main.tex and supplementary.tex.

`s51_verify_claims.py` guards the *numbers*: it fails when a quantity drifts away from
`analysis/results/`. It cannot see the other way a two-document submission goes wrong, which is
that the documents stop agreeing about where things *are*. When a section is split, inserted or
moved, every "Section~VIII of the main paper" in the supplement and every "Supplementary
Table~Sn" in the main paper silently starts pointing somewhere else: LaTeX reports nothing,
because neither document can \\ref into the other. A single restructuring can break a dozen such
pointers at once, which is what this script exists to prevent.

How the two directions are checked:

  supplement -> main   The main paper's numbering is derived from `main.tex` itself, by reading
                       the order of its \\section, table and figure declarations. No compile is
                       needed and no .aux file is consulted, so this half of the check runs on a
                       fresh clone. Every Roman-numeral reference the supplement makes to the
                       main paper is then resolved against that numbering and compared with
                       EXPECT_IN_MAIN below, which records what each reference is *meant* to
                       point at. Reorder the main paper and the assertion fails by name.

  main -> supplement   The supplement's numbering is taken from `supplementary.aux`, because its
                       sections, subsections, tables, figures and listings run on five counters
                       and the compiled values are authoritative. Build the supplement first;
                       without the .aux this half skips and says so.

EXPECT_* are the intent. They are written out in full rather than inferred, because the whole
failure mode is a reference that still resolves to *something* -- a number that exists but
names the wrong object is exactly what a resolver cannot catch on its own.

    python3 analysis/s52_consistency.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')

PROBLEMS, NOTES, SKIPPED = [], [], []

#: every Roman-numeral reference the supplement makes to the main paper, and the label in
#: main.tex it has to land on. Keyed by the numeral as the supplement prints it.
#: Tables are inlined into the documents by analysis/s70_assemble.py, between
#: \tablesource{tables/x.tex} and \tablesourceend{tables/x.tex} (both expand to nothing).
#: tables/ is still the source of truth, so a checker that used to resolve \input matches
#: either form and reads the same file. The markers are macros rather than comments because
#: this module strips full-line comments before it looks.
TABLE_REF = re.compile(r'(?:\\input|\\tablesource)\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}')
TABLE_BLOCK = re.compile(r'\\tablesource\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}'
                         r'.*?\\tablesourceend\{\1\}', re.S)

EXPECT_IN_MAIN = {
    'section': {
        'I': 'sec:introduction',
        'II': 'sec:relatedwork',
        'III': 'sec:method',
        'IV': 'sec:spectrum',
        'V': 'sec:scale',
        'VI': 'sec:scope',
        'VII': 'sec:budget',
        'VIII': 'sec:limitations',
        'IX': 'sec:conclusion',
    },
    'table': {
        #: the claim ledger is Supplementary Table S2, which Section II points at in its
        #: first line, so the body tables are the measurements the paper argues from
        'I': 'tab:spectrum',
        'II': 'tab:bench_main',
        'III': 'tab:recal',
        'IV': 'tab:search',
    },
    'figure': {
        '1': 'fig:spectrum',
        '2': 'fig:controls',
        #: the four body panels of s69_body_panels.py
        '3': 'fig:body',
    },
}

#: every reference main.tex makes into the supplement, and the label it has to land on.
EXPECT_IN_SUPP = {
    'Fig.~S5': 'fig:zc',
    'Fig.~S6': 'fig:sm_controls',
    'Section~S1': 'sec:sm_repro',
    'Section~S2': 'sec:sm_novelty',
    'Section~S4': 'sec:sm_threats',
    'Section~S6': 'sec:sm_fractal',
    'Section~S7': 'sec:sm_moved',
    'Section~S11': 'sec:sm_spaces',
    'Section~S12': 'sec:sm_targets',
    'Section~S13': 'sec:sm_measure',
    'Section~S14': 'sec:sm_policy_full',
    'Section~S16': 'sec:sm_learned',
    'Section~S17': 'sec:sm_proofs',
    'Section~S18': 'sec:sm_robust',
    'Section~S19': 'sec:sm_nats',
    'Table~S2': 'tab:novelty',
    'Table~S8': 'tab:zc_sm',
    'Table~S9': 'tab:zc_spaces',
    'Table~S15': 'tab:bench',
    'Table~S16': 'tab:realization',
    'Table~S13': 'tab:search_sm',
    'Table~S14': 'tab:learned',
}

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII',
         'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']


def read(name):
    with open(os.path.join(ROOT, name)) as fh:
        return fh.read()


# --------------------------------------------------------------------------- main numbering
def main_numbering(tex):
    """label -> printed number for main.tex, from the source order alone.

    IEEEtran numbers sections and tables in Roman and figures in Arabic, and both counters
    advance in the order the declarations appear in the file, so the source is enough. A
    starred section (\\section*) takes no number, which is why it is excluded here -- the Data
    and Code Availability section at the end is one.
    """
    out = {}

    sections = [m for m in re.finditer(r'\\section\{', tex)]
    for i, m in enumerate(sections):
        tail = tex[m.end():m.end() + 400]
        lab = re.search(r'\\label\{([^}]+)\}', tail)
        if lab and i < len(ROMAN):
            out[lab.group(1)] = ROMAN[i]

    # tables: in main.tex every table is pulled in by \input{tables/x.tex}; the label lives in
    # the included file, next to the caption
    for i, m in enumerate(TABLE_REF.finditer(tex)):
        path = os.path.join(ROOT, m.group(1))
        if not os.path.exists(path):
            PROBLEMS.append('main.tex inputs %s, which does not exist' % m.group(1))
            continue
        lab = re.search(r'\\label\{([^}]+)\}', open(path).read())
        if lab and i < len(ROMAN):
            out[lab.group(1)] = ROMAN[i]

    for i, m in enumerate(re.finditer(r'\\begin\{figure\*?\}', tex)):
        tail = tex[m.end():m.end() + 2000]
        lab = re.search(r'\\label\{([^}]+)\}', tail)
        if lab:
            out[lab.group(1)] = str(i + 1)
    return out


# --------------------------------------------------------------------- supplement numbering
def supp_numbering():
    """label -> printed number for the supplement, from its compiled .aux."""
    auxp = os.path.join(ROOT, 'supplementary.aux')
    if not os.path.exists(auxp):
        SKIPPED.append('supplementary.aux is not present, so the main->supplement half of this '
                       'check could not run. Build the supplement first '
                       '(pdflatex supplementary.tex, twice) and rerun.')
        return None
    out = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{(S?[0-9.]+)\}', open(auxp).read()):
        out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------------------ the checks
def check_supp_to_main(main_num, supp):
    """Every Roman reference the supplement makes to the main paper lands where it should."""
    inverse = {v: k for k, v in main_num.items()}
    pats = [
        #: the spacing is \s+ throughout rather than a literal space: LaTeX sources wrap, and
        #: a pointer that happens to straddle a line break is exactly as stale as one that
        #: does not, so a rigid pattern reports a clean run over a reference that is wrong.
        ('section', r'(?:main[- ]paper\s+Section~([IVX]+)'
                    r'|Section~([IVX]+)\s+of\s+the\s+main\s+paper'
                    r"|main\s+paper(?:'s)?,?\s+Section~([IVX]+)"
                    r'|Sections~([IVX]+)\s+and~([IVX]+)\s+of\s+the\s+main\s+paper)'),
        ('table', r'(?:[Mm]ain[- ]paper\s+Table~([IVX]+)'
                  r'|Table~([IVX]+)\s+of\s+the\s+main\s+paper'
                  r"|main\s+paper(?:'s)?,?\s+Table~([IVX]+))"),
        ('figure', r'(?:main[- ]paper\s+Fig\.~([0-9]+)'
                   r'|Fig\.~([0-9]+)\s+of\s+the\s+main\s+paper)'),
    ]
    seen = 0
    for kind, pat in pats:
        for m in re.finditer(pat, supp):
            for num in [g for g in m.groups() if g]:
                seen += 1
                want = EXPECT_IN_MAIN[kind].get(num)
                if want is None:
                    PROBLEMS.append('supplement refers to main-paper %s %s, which this script '
                                    'has no expectation for; add it to EXPECT_IN_MAIN'
                                    % (kind, num))
                    continue
                got = main_num.get(want)
                if got is None:
                    PROBLEMS.append('supplement refers to main-paper %s %s, expected to be '
                                    '\\label{%s}, but main.tex has no such label'
                                    % (kind, num, want))
                elif got != num:
                    PROBLEMS.append('supplement says main-paper %s~%s where \\label{%s} is now '
                                    '%s~%s -- the supplement\'s prose is stale'
                                    % (kind, num, want, kind, got))
                else:
                    NOTES.append('supplement -> main %s %s = %s' % (kind, num, want))
    return seen


def check_compiled_pdfs():
    """Two defects survive every source-level check in this file and still reach a reviewer,
    because they live in the typeset output rather than in the sources.

    The first is an unresolved reference, which LaTeX prints as "??" after writing a warning
    that scrolls past in a quiet build. The second is a length-looking macro set with
    \\setlength: IEEEtran's \\IEEEbibitemsep is a \\def and not a register, so
    \\setlength{\\IEEEbibitemsep}{...} typesets its current value as body text rather than
    changing it, and a stray "0pt plus .5pt" appears above the references.

    Both are caught by reading the compiled PDFs. The check is skipped, not failed, when a PDF
    has not been built or pdftotext is not installed, so `make check` still runs on a bare
    checkout.
    """
    seen = 0
    for name in ('main.pdf', 'supplementary.pdf'):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            SKIPPED.append('%s is not built, so its typeset output was not read' % name)
            continue
        try:
            txt = subprocess.run(['pdftotext', path, '-'], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,
                                 check=True).stdout.decode('utf-8', 'replace')
        except (OSError, subprocess.CalledProcessError):
            SKIPPED.append('pdftotext could not read %s' % name)
            continue
        seen += 1
        n_q = txt.count('??')
        if n_q:
            PROBLEMS.append('%s prints "??" %d time(s): a reference resolves to nothing in '
                            'the typeset output' % (name, n_q))
        #: an undefined \cite prints as [?] rather than ??, which is what a bibtex run that
        #: failed or never happened leaves behind -- a whole document of them, and no
        #: reference list at the end
        n_c = len(re.findall(r'\[\?\]', txt))
        if n_c:
            PROBLEMS.append('%s prints "[?]" %d time(s): a citation resolves to nothing, so '
                            'bibtex did not run or did not finish' % (name, n_c))
        for m in re.finditer(r'\d*\.?\d+pt plus [-.\d]+pt', txt):
            PROBLEMS.append('%s prints the glue "%s" as body text, which is what \\setlength '
                            'on a \\def macro such as \\IEEEbibitemsep does'
                            % (name, m.group(0)))
    return seen


def check_subsection_letters(main, supp):
    """A pointer like "Section~VII-B" names a subsection, and the roman-numeral checks above
    read only the "VII". A section that loses or gains a subsection leaves that letter
    pointing at nothing, which is invisible to every other check here and to LaTeX, since
    the letter is typed rather than referenced. So the letters are counted against the
    subsections main.tex actually declares."""
    #: subsection titles per section, in document order. The titles are kept, not just the
    #: count, because the failure this check exists for comes in two shapes: a letter past
    #: the end of a section, and a letter that still lands inside it but on the wrong
    #: subsection after one was merged away. Only the first can be detected automatically,
    #: so the second is made visible instead -- every resolved pointer is printed with the
    #: title it names, and a wrong one reads as wrong.
    titles, roman = [], ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                         'XI', 'XII']
    for line in main.split('\n'):
        if line.startswith('\\section{'):
            titles.append([])
        elif line.startswith('\\subsection{') and titles:
            titles[-1].append(line[len('\\subsection{'):].rstrip().rstrip('}'))
    by_numeral = {roman[i + 1]: t for i, t in enumerate(titles) if i + 1 < len(roman)}
    seen = 0
    for m in re.finditer(r'Sections?~(?:[IVX]+\s+and~)?([IVX]+)-([A-Z])\b', supp):
        num, letter = m.group(1), m.group(2)
        seen += 1
        have = by_numeral.get(num)
        want = ord(letter) - ord('A') + 1
        if have is None:
            PROBLEMS.append('supplement refers to Section~%s-%s, and main.tex has no '
                            'Section~%s' % (num, letter, num))
        elif want > len(have):
            PROBLEMS.append('supplement refers to Section~%s-%s, but Section~%s of main.tex '
                            'has %d subsection%s -- the letter points at nothing'
                            % (num, letter, num, len(have),
                               '' if len(have) == 1 else 's'))
        else:
            NOTES.append('supplement -> main %s-%s = "%s"'
                         % (num, letter, have[want - 1]))
    return seen


def check_main_to_supp(main, supp_num):
    """Every Sn reference main.tex makes lands on the supplementary object it names."""
    pat = (r'(Section~S[0-9]+(?:\.[0-9]+)?'
           r'|Table~S[0-9]+|Fig\.~S[0-9]+|Listing~S[0-9]+)')
    seen = 0
    for m in re.finditer(pat, main):
        token = m.group(1)
        seen += 1
        want = EXPECT_IN_SUPP.get(token)
        if want is None:
            PROBLEMS.append('main.tex refers to "%s", which this script has no expectation '
                            'for; add it to EXPECT_IN_SUPP' % token)
            continue
        num = token.split('~')[1]
        got = supp_num.get(want)
        if got is None:
            PROBLEMS.append('main.tex refers to %s, expected to be \\label{%s}, but the '
                            'supplement has no such label' % (token, want))
        elif got != num:
            PROBLEMS.append('main.tex says %s where \\label{%s} compiled as %s -- the main '
                            'paper\'s pointer is stale' % (token, want, got))
        else:
            NOTES.append('main -> supplement %s = %s' % (token, want))
    return seen


def labels_in(tex):
    """Every label a document declares, including those a listing sets in its options."""
    return (set(re.findall(r'\\label\{([^}]+)\}', tex))
            | set(re.findall(r'label=\{([^}]+)\}', tex)))


def check_internal_refs(tex, name, own_labels):
    """Every \\ref in a document resolves inside that document."""
    declared = set(own_labels)
    for m in re.finditer(r'\\(?:ref|autoref)\{([^}]+)\}', tex):
        if m.group(1) not in declared:
            PROBLEMS.append('%s has \\ref{%s}, which is declared nowhere in it'
                            % (name, m.group(1)))


def check_no_duplicated_prose(main, supp):
    """A passage moved to the supplement should not also still be in the main paper.

    Checked on long sentences only: a shared sentence of this length is a copy, not a
    coincidence. Captions and the handful of sentences the supplement deliberately requotes
    are allowed through by MOVED_ON_PURPOSE.
    """
    moved_on_purpose = (
        'A free warm start is the cheap version',
        'A shortlist is only as informative as the pool',
        'Fit a surrogate on the genotype before anything else',
        'Treat the prior as a property of the space',
        'Screen free, then allocate epochs by halving',
        'Build a surrogate benchmark out of your own logs',
        'All six directed transfers between the three NAS-Bench-201 datasets',
    )

    def sentences(t):
        t = re.sub(r'%.*', ' ', t)
        t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
        t = re.sub(r'[{}$~\\]', ' ', t)
        t = re.sub(r'\s+', ' ', t)
        return [s.strip() for s in re.split(r'(?<=[.!?]) ', t) if len(s.strip()) > 160]

    a, b = set(sentences(main)), set(sentences(supp))
    for s in sorted(a & b):
        if any(k in s for k in moved_on_purpose):
            continue
        PROBLEMS.append('this sentence is in BOTH main.tex and supplementary.tex: "%s..."'
                        % s[:110])


def check_expectation_map(main_num):
    """EXPECT_IN_MAIN against the numbering main.tex actually produces.

    The map is the fixed point of this whole file: every reference the supplement makes by
    numeral is resolved through it. But only the entries the supplement happens to use are
    exercised by that resolution, so an entry nobody cites can be wrong for a whole revision
    without anything saying so -- and after a renumbering the entries nobody cites are exactly
    the ones most likely to be stale. This checks the map itself, entry by entry.
    """
    for kind, expected in EXPECT_IN_MAIN.items():
        for num, lab in expected.items():
            got = main_num.get(lab)
            if got is None:
                PROBLEMS.append('EXPECT_IN_MAIN says %s %s is %s, but main.tex declares no '
                                'such label' % (kind, num, lab))
            elif got != num:
                PROBLEMS.append('EXPECT_IN_MAIN says %s %s is %s, but main.tex numbers it %s'
                                % (kind, num, lab, got))
    #: and the other direction: a numbered object of a kind the map covers, left out of it,
    #: is a numeral the supplement could cite and this file could not resolve
    for kind, prefix in (('section', 'sec:'), ('table', 'tab:'), ('figure', 'fig:')):
        if kind not in EXPECT_IN_MAIN:
            continue
        listed = set(EXPECT_IN_MAIN[kind].values())
        for lab, num in main_num.items():
            if lab.startswith(prefix) and lab not in listed:
                PROBLEMS.append('main.tex numbers %s %s as %s, and EXPECT_IN_MAIN omits it'
                                % (kind, lab, num))


def main():
    m = read('main.tex')
    s = read('supplementary.tex')
    main_num = main_numbering(m)

    print('main.tex numbering derived from source order')
    for lab, num in sorted(main_num.items(), key=lambda kv: (kv[1].isdigit(), len(kv[1]), kv[1])):
        print('  %-18s %s' % (num, lab))
    print()

    check_expectation_map(main_num)
    n1 = check_supp_to_main(main_num, s)
    supp_num = supp_numbering()
    #: a pointer into the supplement may live in a generated caption rather than in the
    #: prose, and a caption is as able to go stale as a sentence is, so the table files
    #: main.tex \input are checked with it.
    m_with_tables = m
    for mm in TABLE_REF.finditer(m):
        p_ = os.path.join(ROOT, mm.group(1))
        if os.path.exists(p_):
            m_with_tables += '\n' + open(p_).read()
    n2 = check_main_to_supp(m_with_tables, supp_num) if supp_num else 0
    n3 = check_subsection_letters(m, s)

    main_labels = labels_in(m)
    for mm in TABLE_REF.finditer(m):
        p = os.path.join(ROOT, mm.group(1))
        if os.path.exists(p):
            main_labels |= labels_in(open(p).read())
    check_internal_refs(m, 'main.tex', main_labels)

    supp_labels = labels_in(s)
    for mm in TABLE_REF.finditer(s):
        p = os.path.join(ROOT, mm.group(1))
        if os.path.exists(p):
            supp_labels |= labels_in(open(p).read())
    check_internal_refs(s, 'supplementary.tex', supp_labels)

    check_no_duplicated_prose(m, s)
    n4 = check_compiled_pdfs()

    print('resolved %d cross-references from the supplement into the main paper' % n1)
    print('resolved %d cross-references from the main paper into the supplement' % n2)
    #: a subsection pointer that still lands inside its section cannot be checked
    #: automatically, so each one is printed with the subsection it names and read
    subs = [n for n in NOTES if re.match(r'supplement -> main [IVX]+-[A-Z] = ', n)]
    if subs:
        print('subsection pointers, each with the subsection it resolves to:')
        for n in subs:
            print('  %s' % n[len('supplement -> main '):])
    print()
    for p in PROBLEMS:
        print('FAIL  %s' % p)
    for sk in SKIPPED:
        print('NOTE  %s' % sk)
    print('\n%d cross-references checked, %d failed%s'
          % (n1 + n2 + n3 + n4, len(PROBLEMS), ', %d skipped' % len(SKIPPED) if SKIPPED else ''))
    return 1 if PROBLEMS else 0


if __name__ == '__main__':
    sys.exit(main())
