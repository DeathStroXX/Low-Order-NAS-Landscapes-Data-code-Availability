"""Contradictions *inside* the two documents, which the other two checkers cannot see.

`s51_verify_claims.py` checks each registered claim against the result file it came from, and
`s52_consistency.py` checks that pointers between the two documents land on the right object.
Neither can catch the failure mode this script is for: the same thing said twice, in two
places, with two different numbers. A number can agree with its result file in one sentence
and be stale in another, because only one of the two sentences is registered; a duplicated
passage can drift when one copy is revised; and a count written as a word ("two findings",
"nine rows") can stop matching the thing it counts without any number moving at all.

Four families, all mechanical:

  drift        The same sentence in two places with different numbers. Sentences are
               compared with every number masked out, so a passage that was duplicated into
               the supplement and then revised in only one of the two copies collides with
               itself. Matching whole sentences rather than the words around a number is what
               keeps lists of per-dataset values -- "$0.58$, $0.70$ and $0.84$", three numbers
               sharing one run of preceding words -- from reading as a disagreement.
  interval     A value quoted with its own interval, "$x$ $[lo,hi]$", where x is outside
               [lo, hi]. This became possible to get wrong when the search table started
               quoting bootstrap intervals beside every contrast.
  range        A range written "$a$ to $b$" or "$a$--$b$" with a > b. "from $a$ to $b$" is a
               change rather than a range and is skipped, since a quantity is allowed to fall.
  count        A number word immediately before a noun phrase the sources can actually
               count: the propositions declared in the paper, the claims of the ledger, the
               maps of the monotone family, the proxies of the zero-cost family. Only the
               nearest number word counts, so "one of the thirteen proxies" is read as
               thirteen; and only unambiguous phrases are registered, because "tables" and
               bare "maps" mean different things in different sentences and guessing what a
               word counts is how a checker starts lying.
  unmatched    A number in the prose of either document that appears nowhere in
               `analysis/results/`, at any rounding. This is the family with the yield. A
               number that drifted during a revision stops matching the file it came from,
               and unlike `s51_verify_claims.py` this does not need the claim to have been
               registered first -- it sweeps every number in both documents. The exemption
               list names the handful of numbers that legitimately have no result file
               behind them, each with its reason, and anything not on that list is reported.

Everything is read from the sources and from `analysis/results/`; nothing is hard-coded except
the registry and the exemptions, and both are short enough to read.

    python3 analysis/s67_contradictions.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')

PROBLEMS, NOTES = [], []

WORD = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
        'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15}

#: a sentence has to carry this many alphabetic words before it is compared, which is what
#: keeps table rows -- "\quad C10 & 500 & 0.819 & ..." -- out of the drift family
MIN_WORDS = 9


#: Tables are inlined into the documents by analysis/s70_assemble.py, between
#: \tablesource{tables/x.tex} and \tablesourceend{tables/x.tex} (both expand to nothing).
#: tables/ is still the source of truth, so a checker that used to resolve \input matches
#: either form and reads the same file. The markers are macros rather than comments because
#: this module strips full-line comments before it looks.
TABLE_REF = re.compile(r'(?:\\input|\\tablesource)\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}')
TABLE_BLOCK = re.compile(r'\\tablesource\{(tables/[A-Za-z0-9_]+(?:\.tex)?)\}'
                         r'.*?\\tablesourceend\{\1\}', re.S)

def read(path):
    with open(path) as fh:
        return fh.read()


def strip_tex(src):
    """Drop comments and the verbatim blocks, which are shell transcripts rather than prose."""
    src = re.sub(r'(?<!\\)%.*', '', src)
    src = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', ' ', src, flags=re.S)
    src = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', ' ', src, flags=re.S)
    return src


def sources():
    """Every file whose prose a reader sees: the two documents and the tables they pull in."""
    out = {}
    for name in ('main.tex', 'supplementary.tex'):
        raw = read(os.path.join(ROOT, name))
        out[name] = strip_tex(raw)
        for m in TABLE_REF.finditer(raw):
            p = os.path.join(ROOT, m.group(1))
            if os.path.exists(p):
                out[m.group(1)] = strip_tex(read(p))
    return out


# --------------------------------------------------------------- the same sentence, twice
NUM = re.compile(r'(?<![\d.])(\d+(?:\{,\})?\d*(?:\.\d+)?)(?![\d.])')


def sentences(text):
    """Prose sentences, with the markup that does not change meaning taken out."""
    s = re.sub(r'\\(?:cite|ref|eqref|label|input|includegraphics)\{[^}]*\}', ' ', text)
    s = re.sub(r'\\[a-zA-Z]+\*?', ' ', s)
    s = re.sub(r'[{}$~\\]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return [p.strip() for p in re.split(r'(?<=[.!?]) ', s)]


def mask(sent):
    """The sentence with every number replaced by a hash, plus the numbers it carried."""
    nums = [m.group(1).replace('{,}', '') for m in NUM.finditer(sent)]
    body = NUM.sub('#', sent)
    body = re.sub(r'[^a-zA-Z# ]', ' ', body).lower()
    body = re.sub(r'\s+', ' ', body).strip()
    return body, tuple(nums)


def check_drift(src):
    """Sentences whose wording matches but whose numbers do not."""
    seen = {}
    n = 0
    for fname, text in src.items():
        for sent in sentences(text):
            body, nums = mask(sent)
            words = [w for w in body.split() if w != '#' and len(w) > 1]
            if not nums or len(words) < MIN_WORDS:
                continue
            n += 1
            seen.setdefault(body, {}).setdefault(nums, []).append((fname, sent))
    for body, variants in sorted(seen.items()):
        if len(variants) < 2:
            continue
        lines = []
        for nums, places in sorted(variants.items()):
            f = sorted({p[0] for p in places})[0]
            lines.append('      %-22s %s' % (', '.join(nums), f))
        PROBLEMS.append('drift: one sentence, %d different sets of numbers\n      "%s"\n%s'
                        % (len(variants), body[:150], '\n'.join(lines)))
    return n


# ------------------------------------------------------------------ a value inside its interval
IVAL = re.compile(r'\$([+-]?\d+(?:\.\d+)?)\$\s*\$\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*'
                  r'([+-]?\d+(?:\.\d+)?)\s*\]\$')
IVAL_TEX = re.compile(r'([+-]?\d+\.\d+)\$\^\{\\[a-z]+\}\$\s*\\tiny\[\s*([+-]?\d+\.\d+)\s*,\s*'
                      r'([+-]?\d+\.\d+)\s*\]')


def check_intervals(src):
    n = 0
    for fname, text in src.items():
        for rx in (IVAL, IVAL_TEX):
            for m in rx.finditer(text):
                x, lo, hi = (float(g) for g in m.groups())
                n += 1
                if lo > hi:
                    PROBLEMS.append('interval: %s has [%g, %g] with the bounds the wrong way '
                                    'round -- "%s"' % (fname, lo, hi, m.group(0)[:70]))
                elif not (lo - 1e-9 <= x <= hi + 1e-9):
                    PROBLEMS.append('interval: %s quotes %g outside its own interval '
                                    '[%g, %g] -- "%s"' % (fname, x, lo, hi, m.group(0)[:70]))
    return n


RANGE = re.compile(r'\$(\d+(?:\.\d+)?)\$(?:\s*(?:to|--)\s*)\$(\d+(?:\.\d+)?)\$')
RANGE2 = re.compile(r'\$(\d+(?:\.\d+)?)\s*(?:--|\\text\{--\})\s*(\d+(?:\.\d+)?)\$')


def check_ranges(src):
    n = 0
    for fname, text in src.items():
        for rx in (RANGE, RANGE2):
            for m in rx.finditer(text):
                #: "cuts regret from $0.90$ to $0.40$" states a change, and a quantity is
                #: allowed to fall; only a range with no "from" in front of it is a range
                lead = re.sub(r'[^a-z ]', ' ', text[max(0, m.start() - 40):m.start()].lower())
                if 'from' in lead.split():
                    continue
                a, b = float(m.group(1)), float(m.group(2))
                n += 1
                if a > b:
                    PROBLEMS.append('range: %s writes "%s", which runs downwards'
                                    % (fname, re.sub(r'\s+', ' ', m.group(0))))
    return n


# ------------------------------------------------------------------ counts written as words
def registry():
    """noun -> (count, how it was counted). Only nouns the sources can actually count."""
    main = strip_tex(read(os.path.join(ROOT, 'main.tex')))
    reg = {}

    reg = {}
    reg['propositions'] = (len(re.findall(r'\\begin\{proposition\}', main)),
                           'proposition environments in main.tex')
    reg['contributions'] = (3, 'the three the introduction announces')
    try:
        with open(os.path.join(RES, 'novelty.json')) as fh:
            reg['claims'] = (json.load(fh)['n_claims'], 'claims in novelty.json')
    except FileNotFoundError:
        NOTES.append('novelty.json absent; the ledger size is not checked')
    try:
        with open(os.path.join(RES, 'monotone.json')) as fh:
            reg['monotone maps'] = (len(json.load(fh)['family']),
                                    'maps in monotone.json')
    except FileNotFoundError:
        NOTES.append('monotone.json absent; the monotone family size is not checked')
    try:
        with open(os.path.join(RES, 'zc_nb201.json')) as fh:
            reg['zero-cost proxies'] = (len(json.load(fh)['proxies']),
                                        'proxies in zc_nb201.json')
    except FileNotFoundError:
        NOTES.append('zc_nb201.json absent; the proxy count is not checked')
    return reg


def check_counts(src):
    reg = registry()
    words = '|'.join(WORD)
    n = 0
    for fname, text in src.items():
        flat = re.sub(r'\s+', ' ', re.sub(r'\\[a-zA-Z]+\*?', ' ', text))
        for phrase, (want, how) in reg.items():
            #: the nearest number word before the noun, with at most two adjectives between,
            #: so "one of the thirteen proxies" is read as thirteen and not as one
            rx = r'\b(%s)\b((?:\s+[a-zA-Z-]+){0,2}?)\s+%s\b' % (words, re.escape(phrase))
            for m in re.finditer(rx, flat, re.I):
                got = WORD[m.group(1).lower()]
                n += 1
                if got != want:
                    PROBLEMS.append(
                        'count: %s says "%s" but there are %d (%s)'
                        % (fname, re.sub(r'\s+', ' ', m.group(0))[:60], want, how))
    return n


#: prose numbers with no result file behind them, and why each one is allowed. A number that
#: is not here and not in analysis/results/ is reported, which is the point of the family.
UNMATCHED_OK = {
    '17.01': 'quoted from Pushak and Hoos, not computed here',
    '28.58': 'quoted from Pushak and Hoos, not computed here',
    '0.15': 'quoted from Pushak and Hoos, not computed here',
    '4.50': 'quoted from Pushak and Hoos, not computed here',
    '0.24': 'quoted from Pushak and Hoos, not computed here',
    '58.94': 'quoted from Pushak and Hoos, not computed here',
    '93.92': "quoted from NAS-Bench-201's own paper, not computed here",
    '0.30': "quoted from NAS-Bench-201's own paper, not computed here",
    '3.30': 'a difference of two accuracies that results/recipe.json stores separately',
    '5.39': 'a difference of two accuracies that results/recipe.json stores separately',
    '1178': 'a pod-log population size, counted in the log rather than in a result file',
}

PROSE_NUM = re.compile(r'\$([+-]?\d+(?:\{,\})?\d*(?:\.\d+)?)\\?%?\$')


def result_values():
    """Every number in analysis/results/, rendered at every rounding and as a percentage."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            for d in range(7):
                out.add(('%.' + str(d) + 'f') % abs(o))
                out.add(('%.' + str(d) + 'f') % abs(o * 100))

    if not os.path.isdir(RES):
        return None
    for name in sorted(os.listdir(RES)):
        if name.endswith('.json'):
            with open(os.path.join(RES, name)) as fh:
                walk(json.load(fh))
    return out


def check_unmatched(src):
    vals = result_values()
    if vals is None:
        NOTES.append('analysis/results/ absent; prose numbers are not matched against it')
        return 0
    n = 0
    for fname, text in src.items():
        #: the tables are generated from the result files, so s51 already pins them; this
        #: family is for prose, where a number can be typed by hand
        if fname.startswith('tables/'):
            continue
        for m in PROSE_NUM.finditer(text):
            raw = m.group(1).replace('{,}', '')
            n += 1
            bare = raw.lstrip('+-')
            if raw in vals or bare in vals or bare in UNMATCHED_OK:
                continue
            ctx = re.sub(r'\s+', ' ', text[max(0, m.start() - 80):m.end() + 25]).strip()
            PROBLEMS.append('unmatched: %s quotes %s, which is in no result file -- "%s"'
                            % (fname, raw, ctx))
    return n


def main():
    src = sources()
    nd = check_drift(src)
    ni = check_intervals(src)
    nr = check_ranges(src)
    nc = check_counts(src)
    nu = check_unmatched(src)

    print('contradiction scan over %d source files' % len(src))
    print('  prose sentences carrying a number %4d compared for drift' % nd)
    print('  values quoted with an interval  %4d checked' % ni)
    print('  ranges written "a to b"         %4d checked' % nr)
    print('  counts written as a word        %4d checked' % nc)
    print('  prose numbers vs results/       %4d checked' % nu)
    for note in NOTES:
        print('  note: %s' % note)
    print()
    if PROBLEMS:
        for p in PROBLEMS:
            print('FAIL  %s' % p)
        print('\n%d problem(s).' % len(PROBLEMS))
        return 1
    print('no contradiction found by any of the four families.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
