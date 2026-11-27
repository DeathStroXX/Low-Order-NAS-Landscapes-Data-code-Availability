"""Check main.tex against the journal's formal limits, and estimate its printed length.

IEEE Transactions on Neural Networks and Learning Systems, Full Paper:

  * 10 printed pages in two-column IEEEtran format, excluding supplementary material.
    Beyond that a mandatory over-length charge of USD 200 per page applies, and the
    maximum total manuscript length is 15 pages -- a stated cap at this venue, not an
    inferred one. (Brief papers: 6 free, 9 maximum. Survey papers: 15 free, 21 maximum.)
    For scale, published TNNLS full papers run a median of about 15 typeset pages, so a
    paper that stays inside the free allowance is shorter than most of the issue; that is
    a choice, not a defect, and the only thing this script enforces is the cap.
  * abstract: 150 to 250 words, one paragraph, self-contained, and carrying no abbreviations,
    footnotes, references, displayed equations or tables. The bound is the venue's own, so a
    paper outside it fails this script rather than merely being noted.
  * 4 to 5 keywords (index terms).
  * the submission is anonymized (double-anonymous): the submitted PDF must carry no author
    block, so this script fails unless \\blindtrue is set.
  * no impact statement, no mandated section order and no title-length cap. Those are
    requirements of other IEEE titles; checking them here would fail a correct file.

The page estimate is calibrated, not guessed. Two builds of this manuscript were compiled with
the real IEEEtran class and measured with `pdfinfo`, and those two measurements fix the map:

    A: 8546 prose + 427 caption words, 35 table rows, 4 tables, 1 figure, 25 refs -> 10 pages
    B: 9753 prose + 516 caption words, 55 table rows, 5 tables, 1 figure, 64 refs -> 13 pages

Converting each footprint to typeset lines at ~9.5 words per line gives 1098 and 1372 lines,
and the two points fix an affine map from lines to printed pages. This script applies that map
to the current source, so a revision that grows past the limit is caught before a compile and
before submission.

The estimate is an estimate: float packing moves content by a fraction of a page, so treat a
result within 2% of the limit as "verify against the compiled PDF". It is not a substitute for
`pdfinfo main.pdf`, which is authoritative once LaTeX has run.

    python3 analysis/s48_page_budget.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#: Tables are inlined into the documents by analysis/s70_assemble.py, between
#: \tablesource{tables/x.tex} and \tablesourceend{tables/x.tex} (both expand to nothing).
#: tables/ is still the source of truth, so a checker that used to resolve \input matches
#: either form and reads the same file. The markers are macros rather than comments because
#: this module strips full-line comments before it looks.
TABLE_REF = re.compile(r'(?:\\input|\\tablesource)\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}')
TABLE_BLOCK = re.compile(r'\\tablesource\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}'
                         r'.*?\\tablesourceend\{\1\}', re.S)

ROOT = os.path.join(HERE, '..')

#: IEEEtran journal, two columns: words per typeset line.
WORDS_PER_LINE = 9.5
#: printed pages are an affine function of the line count, fitted on compiled IEEEtran builds
#: of this manuscript:
#:     A: 1098 lines -> 10 pages    B: 1250 lines -> 11 pages    C: 1372 lines -> 13 pages
#: The fit uses A and C, whose slope is 1/91.4 lines per page, and the intercept absorbs the
#: title block, the abstract and the column-balancing slack at the end of the bibliography. A
#: single lines-per-page constant fitted on either endpoint alone disagrees with the other by
#: a third of a page, which is why two points are used.
#:
#: B is the point to read this estimate against: at 1250 lines the affine fit says 11.66
#: pages where the compiled PDF is 11. The estimate is therefore an UPPER bound near the free
#: allowance -- floats pack better than a line count can know -- so an estimate of up to about
#: 11.7 can still compile to 11 pages. Compile before paying for a page.
PAGES_PER_LINE = (13.0 - 10.0) / (1372.0 - 1098.0)
PAGES_INTERCEPT = 10.0 - 1098.0 * PAGES_PER_LINE
CALIBRATION = ((1098.0, 10), (1372.0, 13))
#: the middle build, kept for the note above and reported beside the estimate
CALIBRATION_MIDPOINT = (1250.0, 11)
#: a \bibitem under IEEEtran with \IEEEbibitemsep set to 0pt occupies about this many lines
LINES_PER_REF = 2.9
#: fixed overhead per float: rules, caption leading, the \addlinespace rows of a booktabs table
LINES_PER_TABLE = 4.0
#: A numeric table occupies one typeset line per row, which is what the two calibration builds
#: contained. A table whose cells hold sentences does not: its text wraps inside the column and
#: the float grows well past its row count, so the row count alone is not a safe estimate. We
#: therefore charge each table the greater of its row count and its text volume, converting the
#: volume at the rate below: the \footnotesize these tables use fits roughly 1.25x more words
#: per line than body text.
#:
#: A table* is charged at the same word rate rather than half of it. Its lines are twice as
#: wide, so it needs half as many -- but each costs two body lines of page area, and this
#: estimate is denominated in body lines. For the same reason its row count is doubled.
#:
#: The surcharge applies only above TEXT_TABLE_RATIO, and that threshold is what keeps the
#: calibration valid. Both calibration builds were measured with the plain row-count rule, and
#: every table in them is numeric: the densest, the proxy comparison, comes out at 1.75x its row
#: count, comfortably under the threshold, so each is still charged what it was charged when the
#: affine map was fitted. Only a table shaped like prose -- more than twice as much cell text as
#: rows -- is charged for its text instead. Lowering this threshold would mean refitting the
#: calibration against the two compiled PDFs.
TABLE_WORDS_PER_LINE = WORDS_PER_LINE * 1.25
#: a table* holds twice the words per line and costs twice the area per line, so the two
#: factors cancel and the rate is the same as for a single-column table
TABLE_WORDS_PER_LINE_WIDE = TABLE_WORDS_PER_LINE
TEXT_TABLE_RATIO = 2.0
#: a single-column \includegraphics[width=\linewidth] figure plus its caption
LINES_PER_FIGURE = 30.0
#: A figure* spans both columns, so a panel row of it occupies twice the typeset area of the
#: same row in one column, and its caption is set across the full measure; charging it at the
#: single-column rate understates it by about that factor. The calibration points contain only
#: single-column figures, so this surcharge does not disturb them.
LINES_PER_FIGURE_WIDE = 60.0

#: Set these from the target journal's current author guide before submission. Both values
#: below are stated on this venue's own page charges page: PAGE_LIMIT is the free allowance,
#: pages beyond it are billed at OVERLENGTH_CHARGE_USD each, and HARD_LIMIT is the maximum
#: total manuscript length, excluding supplementary material, that the venue will accept.
JOURNAL = 'IEEE Transactions on Neural Networks and Learning Systems'
SUBMISSION_TYPE = 'Full Paper'
PAGE_LIMIT = 10
HARD_LIMIT = 15
OVERLENGTH_CHARGE_USD = 200
#: The venue states the bound itself -- 150 to 250 words, one paragraph -- so this is a
#: requirement rather than a house guard, and the report says which it is.
ABSTRACT_MIN, ABSTRACT_MAX = 150, 250
ABSTRACT_LIMIT_IS_VENUE_RULE = True
KEYWORDS_MIN, KEYWORDS_MAX = 4, 5
#: This venue requires no impact statement and caps neither the title nor the section order.
#: The three checks below are therefore switched off rather than deleted, so that re-aiming the
#: paper at a title that does require them is one edit.
REQUIRE_IMPACT = False
IMPACT_MIN, IMPACT_MAX = 100, 150
TITLE_MAX_WORDS = None
TITLE_BANNED = ()
#: The submission is anonymized (double-anonymous), so the submitted file must withhold the
#: author block: \\blindtrue.
REQUIRE_BLIND = True

PROBLEMS, NOTES = [], []


def _words(tex):
    """Word count of TeX prose: macros, math delimiters and braces do not print as words."""
    return len(re.sub(r'\\[a-zA-Z]+\*?|[${}]|~', ' ', tex).split())


def main():
    src = open(os.path.join(ROOT, 'main.tex')).read()
    body = re.sub(r'(?m)^\s*%.*$', '', src)

    # ---------------- abstract ----------------
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S)
    if not m:
        PROBLEMS.append('no abstract found')
        abstract = ''
    else:
        abstract = m.group(1)
        n = _words(abstract)
        _why = ('the journal requires' if ABSTRACT_LIMIT_IS_VENUE_RULE
                else 'this venue states no numeric limit; the house guard is')
        if not ABSTRACT_MIN <= n <= ABSTRACT_MAX:
            PROBLEMS.append('abstract is %d words; %s %d-%d'
                            % (n, _why, ABSTRACT_MIN, ABSTRACT_MAX))
        else:
            NOTES.append('abstract: %d words (%s %d-%d)'
                         % (n, _why, ABSTRACT_MIN, ABSTRACT_MAX))
        if re.search(r'\\cite\{', abstract):
            PROBLEMS.append('abstract contains a citation; the journal forbids references there')
        if re.search(r'\\begin\{(equation|tabular|table)', abstract):
            PROBLEMS.append('abstract contains a displayed equation or table')
        if abstract.strip().count('\n\n'):
            PROBLEMS.append('abstract is more than one paragraph')
        # an abbreviation is a run of >=2 capitals that is not a benchmark/dataset proper noun
        allowed = {'CIFAR', 'NAS', 'SVHN', 'ILSVRC'}
        plain = re.sub(r'\\[a-zA-Z]+\*?', ' ', abstract)
        abbr = {w for w in re.findall(r'\b[A-Z]{2,}\b', plain)} - allowed
        if abbr:
            PROBLEMS.append('abstract contains abbreviations the journal forbids: %s'
                            % ', '.join(sorted(abbr)))
        else:
            NOTES.append('abstract: no abbreviations')

    # ---------------- keywords ----------------
    m = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', body, re.S)
    if not m:
        PROBLEMS.append('no IEEEkeywords block found')
    else:
        k = [x for x in (y.strip() for y in m.group(1).split(',')) if x]
        if not KEYWORDS_MIN <= len(k) <= KEYWORDS_MAX:
            PROBLEMS.append('%d keywords; the journal requires %d-%d'
                            % (len(k), KEYWORDS_MIN, KEYWORDS_MAX))
        else:
            NOTES.append('keywords: %d (%d-%d required)'
                         % (len(k), KEYWORDS_MIN, KEYWORDS_MAX))

    # ---------------- impact statement ----------------
    m = re.search(r'\\section\*\{Impact Statement\}(.*?)(?=\\section|\\begin\{figure|$)',
                  body, re.S)
    impact = ''
    if not m:
        if REQUIRE_IMPACT:
            PROBLEMS.append('no "Impact Statement" section; the journal requires one of '
                            '%d-%d words after the abstract and may return the paper without '
                            'review without it' % (IMPACT_MIN, IMPACT_MAX))
        else:
            NOTES.append('impact statement: not required by this venue, and absent')
    else:
        impact = m.group(1)
        n = _words(impact)
        if not IMPACT_MIN <= n <= IMPACT_MAX:
            PROBLEMS.append('impact statement is %d words; the journal requires %d-%d'
                            % (n, IMPACT_MIN, IMPACT_MAX))
        else:
            NOTES.append('impact statement: %d words (%d-%d)' % (n, IMPACT_MIN, IMPACT_MAX))

    # ---------------- title ----------------
    m = re.search(r'\\title\{(.*?)\}\s*\n', body, re.S)
    if m:
        t = re.sub(r'\\\\', ' ', m.group(1))
        n = _words(t)
        if TITLE_MAX_WORDS is None:
            NOTES.append('title: %d words (this venue states no cap)' % n)
        elif n > TITLE_MAX_WORDS:
            PROBLEMS.append('title is %d words; the journal caps it at %d'
                            % (n, TITLE_MAX_WORDS))
        else:
            NOTES.append('title: %d words (limit %d)' % (n, TITLE_MAX_WORDS))
        bad = [w for w in TITLE_BANNED if re.search(r'\\b%s\\b' % w, t, re.I)]
        if bad:
            PROBLEMS.append('title uses %s; the author guide asks that it be avoided'
                            % ', '.join(bad))

    # ---------------- anonymity ----------------
    _blinded = bool(re.search(r'(?m)^\\blindtrue\s*$', body))
    if REQUIRE_BLIND:
        if _blinded:
            NOTES.append('double-anonymous: \\blindtrue is set')
        else:
            PROBLEMS.append('review is double-anonymous but \\blindtrue is not set: the '
                            'submitted PDF would carry the author names')
    else:
        if _blinded:
            PROBLEMS.append('review is single-anonymous but \\blindtrue is set: the '
                            'submitted PDF would withhold the author block the reviewers '
                            'are given. Set \\blindfalse.')
        else:
            NOTES.append('single-anonymous: \\blindfalse is set, so the author block ships')

    # ---------------- footprint ----------------
    prose = body[body.index(r'\section{Introduction'):body.index(r'\bibliography')]
    #: An inlined table is not prose. Strip the whole marker block, not just a command.
    text_words = (_words(TABLE_BLOCK.sub('', re.sub(r'\\input\{[^}]*\}', '', prose)))
                  + _words(impact))

    cap_words, rows, tables = 0, 0, 0
    for name in [m.split('/', 1)[1] for m in TABLE_REF.findall(prose)]:
        path = os.path.join(ROOT, 'tables', name if name.endswith('.tex') else name + '.tex')
        if not os.path.exists(path):
            PROBLEMS.append('main.tex inputs tables/%s, which does not exist' % name)
            continue
        tbl = open(path).read()
        tables += 1
        cap = re.search(r'\\caption\{(.*?)\}\s*\n\\label', tbl, re.S)
        if cap:
            cap_words += _words(cap.group(1))
        body_rows = tbl.count(r'\\')
        #: the table body with the preamble, caption and label stripped out, so what is left
        #: is the text the reader actually sees inside the cells
        inner = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', tbl, re.S)
        wide = r'\begin{table*}' in tbl
        if inner:
            cells = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ',
                           inner.group(0))
            per = TABLE_WORDS_PER_LINE_WIDE if wide else TABLE_WORDS_PER_LINE
            text_rows = _words(cells) / per
        else:
            text_rows = 0.0
        if text_rows <= TEXT_TABLE_RATIO * body_rows:
            text_rows = 0.0
        else:
            NOTES.append('tables/%s is prose-shaped: charged %.0f lines for its cell text '
                         'rather than %d for its %d rows'
                         % (name, text_rows, body_rows, body_rows))
        charged = max(body_rows, text_rows)
        if wide:
            # a full-width float occupies both columns, so each of its lines is two body lines
            charged = max(2.0 * body_rows, text_rows)
        rows += charged

    wide_figures = len(re.findall(r'\\begin\{figure\*\}', prose))
    figures = len(re.findall(r'\\begin\{figure', prose)) - wide_figures
    figure_lines = LINES_PER_FIGURE * figures + LINES_PER_FIGURE_WIDE * wide_figures
    for fm in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', prose):
        fig = fm.group(1)
        cands = [os.path.join(ROOT, fig), os.path.join(ROOT, fig + '.pdf')]
        if not any(os.path.exists(c) for c in cands):
            PROBLEMS.append('main.tex includes %s, which does not exist' % fig)

    # Citations live in main.tex and in the generated tables it \input's; bibtex sees both, so
    # a count taken from main.tex alone disagrees with main.bbl by whatever the tables cite.
    cited = set()
    cite_src = body
    for rel in TABLE_REF.findall(body):
        tp = os.path.join(ROOT, rel if rel.endswith('.tex') else rel + '.tex')
        if os.path.exists(tp):
            cite_src += '\n' + open(tp).read()
    for cm in re.finditer(r'\\cite\{([^}]*)\}', cite_src):
        cited.update(x.strip() for x in cm.group(1).split(','))
    bbl = os.path.join(ROOT, 'main.bbl')
    if os.path.exists(bbl):
        refs = open(bbl).read().count(r'\bibitem')
        if refs != len(cited):
            PROBLEMS.append('main.bbl has %d entries but main.tex cites %d distinct keys; '
                            'rerun bibtex or prune the .bbl' % (refs, len(cited)))
        else:
            NOTES.append('bibliography: %d entries, all cited' % refs)
    else:
        # The paper uses \bibliography{bibmain}, so there are no \bibitem lines to count
        # until bibtex has run.  Counting the cited keys instead is what keeps the page
        # estimate honest: at ~2.9 lines each, pretending the bibliography is empty
        # understates the paper by more than a page.
        refs = len(cited)
        NOTES.append('bibliography: main.bbl not built; charging %d cited keys '
                     '(run bibtex for the authoritative count)' % refs)

    lines = ((text_words + cap_words) / WORDS_PER_LINE
             + rows + LINES_PER_TABLE * tables
             + figure_lines
             + LINES_PER_REF * refs)
    pages = PAGES_PER_LINE * lines + PAGES_INTERCEPT

    print('footprint of main.tex')
    print('  prose words          %6d' % text_words)
    print('  caption words        %6d' % cap_words)
    print('  table lines charged  %6d  (in %d tables)' % (round(rows), tables))
    print('  figures              %6d  (%d single-column, %d full-width at %.0f lines each)'
          % (figures + wide_figures, figures, wide_figures, LINES_PER_FIGURE_WIDE))
    print('  bibliography entries %6d' % refs)
    print('  estimated lines      %6d  (calibration: %d lines -> %d pages, %d -> %d)'
          % (round(lines), CALIBRATION[0][0], CALIBRATION[0][1],
             CALIBRATION[1][0], CALIBRATION[1][1]))
    print('  estimated pages      %6.2f  (limit %d, hard cap %d)'
          % (pages, PAGE_LIMIT, HARD_LIMIT))
    print('  upper bound: a compiled build at %d lines came out at %d pages, where '
          'this fit says %.2f' % (CALIBRATION_MIDPOINT[0], CALIBRATION_MIDPOINT[1],
                                  PAGES_INTERCEPT + CALIBRATION_MIDPOINT[0] * PAGES_PER_LINE))

    if pages > HARD_LIMIT:
        PROBLEMS.append('estimated %.2f pages exceeds the journal\'s hard cap of %d'
                        % (pages, HARD_LIMIT))
    elif pages > PAGE_LIMIT * 1.02:
        # Exceeding the free allowance is not a defect at this venue -- the median published
        # regular paper does it -- so this is priced, not failed.  Only HARD_LIMIT fails.
        over = int(pages - PAGE_LIMIT + 0.999)
        NOTES.append('estimated %.2f pages: about %d page(s) beyond the %d-page free '
                     'allowance, USD %d in mandatory over-length charges (normal here; '
                     'budget for it)'
                     % (pages, over, PAGE_LIMIT, over * OVERLENGTH_CHARGE_USD))
    elif pages > PAGE_LIMIT:
        NOTES.append('estimated %.2f pages: within 2%% of the limit, so confirm against the '
                     'compiled PDF (pdfinfo main.pdf) before submitting' % pages)
    else:
        NOTES.append('estimated %.2f pages: inside the %d-page limit' % (pages, PAGE_LIMIT))

    pdf = os.path.join(ROOT, 'main.pdf')
    if os.path.exists(pdf):
        NOTES.append('main.pdf is present; `pdfinfo main.pdf` is authoritative over the '
                     'estimate above')

    print()
    for n in NOTES:
        print('  ok   %s' % n)
    for p in PROBLEMS:
        print('  FAIL %s' % p)
    print('\n%d checks passed, %d failed' % (len(NOTES), len(PROBLEMS)))

    out = {
        'journal': JOURNAL,
        'submission_type': SUBMISSION_TYPE,
        'limits': {'pages': PAGE_LIMIT, 'hard_cap_pages': HARD_LIMIT,
                   'overlength_charge_usd_per_page': OVERLENGTH_CHARGE_USD,
                   'abstract_words': [ABSTRACT_MIN, ABSTRACT_MAX],
                   'keywords': [KEYWORDS_MIN, KEYWORDS_MAX]},
        'calibration_lines_to_pages': {'points': [list(CALIBRATION[0]), list(CALIBRATION[1])],
                                       'pages_per_line': PAGES_PER_LINE,
                                       'intercept': PAGES_INTERCEPT},
        'footprint': {'prose_words': text_words, 'caption_words': cap_words,
                      'table_lines': round(rows, 1), 'tables': tables, 'figures': figures,
                      'references': refs, 'estimated_lines': round(lines, 1),
                      'estimated_pages': round(pages, 2)},
        'abstract_words': _words(abstract),
        'notes': NOTES, 'problems': PROBLEMS,
    }
    res = os.path.join(HERE, 'results')
    if os.path.isdir(res):
        with open(os.path.join(res, 'page_budget.json'), 'w') as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write('\n')
        print('wrote analysis/results/page_budget.json')
    return 1 if PROBLEMS else 0


if __name__ == '__main__':
    sys.exit(main())
