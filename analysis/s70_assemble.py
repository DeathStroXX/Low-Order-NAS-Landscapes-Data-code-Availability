"""Inline the generated tables into main.tex and supplementary.tex.

The two documents are submitted as one file each: TNNLS and Overleaf both take a single
source far more gracefully than a tree of \\input, and a missing tables/ directory is the
most common way a transferred project fails to build. The generated tables are still
written to tables/ by the scripts that compute them -- that stays the source of truth --
and this script copies each one into the document between a pair of markers:

    \\tablesource{tables/tab_spectrum.tex}
    ... the table ...
    \\tablesourceend{tables/tab_spectrum.tex}

The markers are LaTeX macros, not comments, and each document's preamble declares them to
expand to nothing. They have to survive comment stripping, because the checkers that resolve
"which table is this" -- s48, s51, s52, s67 -- drop full-line comments before they look, and
a marker they cannot see is a table they would charge nothing for.

It is idempotent. On its first run it converts every \\input{tables/...} into such a block;
on every run after that it replaces the body of each block with the current file. Edit the
table in tables/ and re-run; never edit the inlined copy, because the next run overwrites it.

    python3 analysis/s70_assemble.py            # refresh both documents
    python3 analysis/s70_assemble.py --check    # fail if any block is stale (for `make check`)
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = ['main.tex', 'supplementary.tex']

NAME = r'(tables/[A-Za-z0-9_]+(?:\.tex)?)'
#: a table pulled in by \input, or already inlined and marked
INPUT_LINE = re.compile(r'^[ \t]*\\input\{' + NAME + r'\}[ \t]*$', re.M)
INPUT_ANY = re.compile(r'\\input\{' + NAME + r'\}')
BLOCK = re.compile(r'\\tablesource\{' + NAME + r'\}.*?\\tablesourceend\{\1\}', re.S)
#: the comment-marker form this script emitted before the macros replaced it
OLD_BLOCK = re.compile(r'%%>>> ' + NAME + r'.*?%%<<< \1', re.S)

DECL = ('%% The generated tables are inlined below by analysis/s70_assemble.py, each between\n'
        '%% \\tablesource{...} and \\tablesourceend{...}. Both expand to nothing; they exist so\n'
        '%% the checks in analysis/ can still tell which table is which. Edit tables/, not here.\n'
        '\\providecommand{\\tablesource}[1]{}\n'
        '\\providecommand{\\tablesourceend}[1]{}\n')


def _rel(name):
    return name if name.endswith('.tex') else name + '.tex'


def _block(name):
    rel = _rel(name)
    with open(os.path.join(ROOT, rel)) as fh:
        body = fh.read().rstrip('\n')
    return '\\tablesource{%s}\n%s\n\\tablesourceend{%s}' % (rel, body, rel)


def assemble(text):
    """Return (new_text, [tables inlined])."""
    done = []

    def sub(m):
        done.append(_rel(m.group(1)))
        return _block(m.group(1))

    text = OLD_BLOCK.sub(sub, text)       # migrate the earlier comment markers
    text = BLOCK.sub(sub, text)           # refresh what is already inlined
    text = INPUT_LINE.sub(sub, text)      # a table on a line of its own
    text = INPUT_ANY.sub(sub, text)       # and one pulled in mid-line, inside \resizebox

    done = [_rel(m.group(1)) for m in BLOCK.finditer(text)]
    if done and '\\providecommand{\\tablesource}' not in text:
        i = text.index('\\begin{document}')
        text = text[:i] + DECL + '\n' + text[i:]
    return text, done


def main():
    check = '--check' in sys.argv
    stale = []
    for doc in DOCS:
        p = os.path.join(ROOT, doc)
        with open(p) as fh:
            old = fh.read()
        new, done = assemble(old)
        fresh = new == old
        if not check and not fresh:
            with open(p, 'w') as fh:
                fh.write(new)
        if check and not fresh:
            stale.append(doc)
        print('%-20s %2d tables inlined%s' % (
            doc, len(done), '' if not check else ('  up to date' if fresh else '  STALE')))
    if stale:
        raise SystemExit('s70: %s stale; run `python3 analysis/s70_assemble.py`'
                         % ', '.join(stale))


if __name__ == '__main__':
    main()
