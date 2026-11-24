#!/usr/bin/env python3
"""The exact spectrum on the two remaining complete factorials, from the reader's own tables.

Section IV computes the amplitude spectrum exactly wherever the space is a complete factorial
with one observation per cell, and `s53` and `s57` do that for NAS-Bench-201 ($5^6$) and
NAS-Bench-Macro ($3^8$). Two further public benchmarks satisfy the same condition:

  * **NATS-Bench, size search space (SSS)** -- Dong et al. Five searchable layers, eight
    candidate channel widths each (8, 16, 24, 32, 40, 48, 56, 64), so a complete
    $8^5 = 32{,}768$ factorial, tabulated on CIFAR-10, CIFAR-100 and ImageNet16-120. It tests
    the concentration claim on a *size* space rather than an operation space: the factors are
    ordinal widths, not categorical operations, which is the one structural difference that
    could plausibly change where the variance sits.

  * **TransNAS-Bench-101, micro (cell) space** -- Duan et al., CVPR 2021. Six cell edges, four
    operations each, so a complete $4^6 = 4{,}096$ factorial, tabulated on seven tasks that
    differ in *type* rather than in difficulty (classification, jigsaw, room layout, surface
    normal, semantic segmentation, autoencoding). Every other benchmark here varies the
    difficulty of one task type; this one is the only way to ask whether the spectrum is a
    property of the search space or of the objective.

Neither table is redistributed in this archive -- see `data/external/PROVENANCE_EXTRA.md` for
what each one is, who publishes it, and the route to it -- so this script is a pipeline rather
than a result. It looks for an export of each table under `data/external/`, and:

  * if a table is absent it prints the fetch and export instruction for it and skips that grid.
    Nothing is written for a grid that was not computed, and a run in which both are absent
    writes no file at all: a placeholder in `analysis/results/` would be worse than its absence.
  * if a table is present it verifies that the design really is the complete factorial claimed,
    covered exactly once, refusing anything else rather than imputing or averaging; then runs
    the project's own decomposition (`anova.to_grid`, `anova.check_consistency`,
    `anova.spectrum`), which is the same Moebius inversion `s53` and `s57` use, specialised to
    nothing.

Because the two benchmarks are served as a NATS-Bench pickle and a PyTorch checkpoint, neither
of which this archive can read with numpy and pandas alone, the expected input is an explicit,
documented plain-text export the reader makes from their own copy with the authors' API. The
expectation is stated in `EXPECTED_FORMAT` below, which the script prints verbatim when a file
is present but cannot be read as described.

Expected inputs (either may be overridden by an environment variable):

  data/external/nats-bench-sss.csv            NATS_SSS_PATH
  data/external/transnas-bench-101-micro.csv  TRANSNAS_PATH

Writes, only if at least one grid was computed:
  analysis/results/extra_grids.json   per-order and cumulative shares of every response, the
                                      analytic white-noise null of the grid shape, the skewness
                                      and top-decile spread of every response, and the
                                      provenance of the file each came from
  tables/tab_extra.tex                the same, in the style of tables/tab_bench.tex

    python3 analysis/s63_extra_grids.py
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

#: `python3 -I` does not put the script's own directory on the path, so the sibling modules of
#: this package are made importable explicitly; the script then runs identically from the
#: project root and from inside `analysis/`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anova  # noqa: E402
#: the compression summary of Section VI, reused rather than restated: `s57` already defines
#: skewness and the top-decile spread for a complete grid, and importing it keeps the two
#: benchmarks' numbers comparable by construction. Importing the module runs no analysis.
from s57_benchmarks import compression  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(HERE, 'results')
TBL = os.path.join(ROOT, 'tables')
EXT = os.path.join(ROOT, 'data', 'external')

#: NATS-Bench SSS: the eight candidate channel widths, in the order the benchmark lists them
SSS_CHANNELS = [8, 16, 24, 32, 40, 48, 56, 64]
#: TransNAS-Bench-101 micro: four operations per edge, as 0..3 in the benchmark's own encoding
TRANSNAS_OPS = 4

EXPECTED_FORMAT = """\
Expected input format
---------------------
Both tables are read as a plain export the reader makes from their own copy of the benchmark
with the authors' API. Two shapes are accepted, chosen by the file suffix.

CSV (.csv) -- one header row, one row per architecture, no index column:

  * one design column, either
        arch   the architecture's own identifier as the benchmark prints it:
                 NATS-Bench SSS    the five channel widths, separated by any non-digit, e.g.
                                   8:16:32:48:64 ; each width one of 8 16 24 32 40 48 56 64
                 TransNAS micro    the six cell operations, either as the benchmark's full
                                   name 64-41414-1_02_333 or as the bare six digits 102333 ;
                                   each digit in 0..3
    or
        f0 f1 ... f{d-1}   one integer column per factor, holding 0-based level indices
                           (f0..f4 for the 8^5 grid, f0..f5 for the 4^6 grid)
    If both are present, `arch` is used and the f-columns are ignored.
  * every remaining column must be numeric. Each one is treated as a response and decomposed
    separately, under the name the header gives it. The expected sets are
        NATS-Bench SSS    acc_cifar10, acc_cifar100, acc_ImageNet16-120   (test accuracy, %)
        TransNAS micro    one column per task, i.e. cls_object, cls_scene, autoencoder,
                          normal, sem_segment, room_layout, jigsaw, each holding that task's
                          own primary metric on whichever split you exported
    Any other naming is accepted and carried through; nothing is renamed or rescaled here.
  * a row count equal to the number of cells, and every cell covered exactly once:
    8^5 = 32768 rows for NATS-Bench SSS, 4^6 = 4096 for TransNAS-Bench-101 micro. A missing,
    duplicated or out-of-range cell is refused, not imputed.

JSON (.json) -- a single object mapping the architecture identifier to an object of response
name to number, with the same identifier syntax and the same completeness requirement:

  {"8:16:32:48:64": {"acc_cifar10": 93.12, "acc_cifar100": 70.44,
                     "acc_ImageNet16-120": 45.31}, ...}

Rows whose response value is missing are refused rather than dropped: the decomposition is
exact only on a complete design, so a partial table is a different measurement and this script
will not silently make it.\
"""


def sss_genotype(s):
    """The five channel widths of a NATS-Bench SSS architecture, as level indices."""
    toks = [t for t in re.split(r'[^0-9]+', str(s).strip()) if t]
    if len(toks) != 5:
        raise ValueError('expected five channel widths, got %r' % (s,))
    try:
        return [SSS_CHANNELS.index(int(t)) for t in toks]
    except ValueError:
        raise ValueError('channel width outside %s in %r' % (SSS_CHANNELS, s))


def transnas_genotype(s):
    """The six cell operations of a TransNAS-Bench-101 micro architecture, as level indices."""
    t = str(s).strip()
    if '-' in t:
        t = t.rsplit('-', 1)[1]
    digits = [c for c in t if c.isdigit()]
    if len(digits) != 6:
        raise ValueError('expected a six-operation cell code, got %r' % (s,))
    g = [int(c) for c in digits]
    if min(g) < 0 or max(g) >= TRANSNAS_OPS:
        raise ValueError('operation index outside 0..%d in %r' % (TRANSNAS_OPS - 1, s))
    return g


#: Each grid: where to look, what shape it must be, how to read an architecture identifier, the
#: responses the export is expected to carry, and what to tell the reader if it is not there.
GRIDS = [
    dict(key='nats_sss',
         name='NATS-Bench size search space (SSS)',
         source='Dong et al., NATS-Bench: Benchmarking NAS Algorithms for Architecture '
                'Topology and Size',
         tex='NATS-Bench size search space, $8^{5}=32{,}768$ cells',
         env='NATS_SSS_PATH',
         default='nats-bench-sss.csv',
         shape=(8,) * 5,
         levels=SSS_CHANNELS,
         genotype=sss_genotype,
         expected=['acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'],
         note='five searchable layers, eight candidate channel widths each',
         fetch=[
             'NATS-Bench is published by Dong et al. and its benchmark files are distributed',
             'by the authors\' repository, from the links that repository gives:',
             '    https://github.com/D-X-Y/NATS-Bench',
             'Take the current download link for the *size* search space (the files named',
             'NATS-sss-...) from that repository; this archive deliberately does not hard-code',
             'a link it cannot check. Then export the complete 8^5 grid from your own copy',
             'with the authors\' nats_bench API -- one row per architecture, the five channel',
             'widths in the `arch` column and one accuracy column per dataset -- and save it',
             'as data/external/nats-bench-sss.csv (or point NATS_SSS_PATH at it).',
         ]),
    dict(key='transnas_micro',
         name='TransNAS-Bench-101 micro (cell) space',
         source='Duan et al., TransNAS-Bench-101: Improving Transferability and '
                'Generalizability of Cross-Task Neural Architecture Search, CVPR 2021',
         tex='TransNAS-Bench-101 micro space, $4^{6}=4{,}096$ cells',
         env='TRANSNAS_PATH',
         default='transnas-bench-101-micro.csv',
         shape=(TRANSNAS_OPS,) * 6,
         levels=list(range(TRANSNAS_OPS)),
         genotype=transnas_genotype,
         expected=['cls_object', 'cls_scene', 'autoencoder', 'normal', 'sem_segment',
                   'room_layout', 'jigsaw'],
         note='six cell edges, four operations each, over seven task types',
         fetch=[
             'TransNAS-Bench-101 is published by Duan et al. (CVPR 2021). Its authors serve the',
             'benchmark file themselves, on request or after a short registration, rather than',
             'as an open download, so there is no stable URL to quote here: follow the link',
             'the paper and its project page give and accept the terms stated there.',
             'Then export the complete 4^6 micro grid from your own copy with the authors\'',
             'API -- one row per architecture, the six-operation cell code in the `arch`',
             'column and one column per task holding that task\'s own primary metric -- and',
             'save it as data/external/transnas-bench-101-micro.csv (or point TRANSNAS_PATH',
             'at it).',
         ]),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def resolve(g):
    """Where this grid's table is expected, honouring the environment override."""
    env = os.environ.get(g['env'], '').strip()
    if env:
        return env, True
    return os.path.normpath(os.path.join(EXT, g['default'])), False


def shown(path, used_env):
    """The path as the reader should see it: relative to the project root when it is inside."""
    return path if used_env else os.path.relpath(path, os.path.abspath(ROOT))


class FormatError(Exception):
    """The file is there but is not the export this script documents."""


def read_table(path, g):
    """The export as a (G, {response: values}) pair. Raises FormatError with a reason."""
    suffix = os.path.splitext(path)[1].lower()
    try:
        if suffix == '.json':
            with open(path) as fh:
                obj = json.load(fh)
            if not isinstance(obj, dict) or not obj:
                raise FormatError('the JSON file is not a non-empty object of '
                                  'architecture -> {response: number}')
            keys = sorted(obj)
            if not all(isinstance(obj[k], dict) for k in keys):
                raise FormatError('every value of the JSON object must itself be an object of '
                                  'response name -> number')
            df = pd.DataFrame([obj[k] for k in keys])
            df.insert(0, 'arch', keys)
        elif suffix in ('.csv', '.tsv', '.txt'):
            df = pd.read_csv(path, sep='\t' if suffix == '.tsv' else ',')
        else:
            raise FormatError('unrecognised suffix %r: expected .csv, .tsv or .json' % suffix)
    except FormatError:
        raise
    except Exception as exc:                       # malformed file, bad encoding, bad JSON
        raise FormatError('%s: %s' % (type(exc).__name__, exc))

    d = len(g['shape'])
    fcols = ['f%d' % i for i in range(d)]
    if 'arch' in df.columns:
        design, drop = 'arch', ['arch'] + [c for c in fcols if c in df.columns]
        try:
            G = np.array([g['genotype'](a) for a in df['arch']], dtype=int)
        except ValueError as exc:
            raise FormatError('could not read the `arch` column: %s' % exc)
    elif all(c in df.columns for c in fcols):
        design, drop = ','.join(fcols), fcols
        try:
            G = df[fcols].to_numpy(dtype=int)
        except Exception as exc:
            raise FormatError('the %s columns must be integers: %s' % (','.join(fcols), exc))
        for i, q in enumerate(g['shape']):
            if G[:, i].min() < 0 or G[:, i].max() >= q:
                raise FormatError('column f%d holds level indices outside 0..%d' % (i, q - 1))
    else:
        raise FormatError('no design column: expected `arch`, or the %d columns %s'
                          % (d, ', '.join(fcols)))

    resp = {}
    for c in df.columns:
        if c in drop:
            continue
        v = pd.to_numeric(df[c], errors='coerce')
        if v.isna().all():
            raise FormatError('column %r is not numeric; every column other than the design '
                              'column is read as a response' % c)
        if v.isna().any():
            raise FormatError('column %r has %d missing or non-numeric values; the '
                              'decomposition is exact only on a complete design, so a partial '
                              'column is refused rather than filled in'
                              % (c, int(v.isna().sum())))
        resp[c] = v.to_numpy(float)
    if not resp:
        raise FormatError('no response column: the file carries the design and nothing else')
    return G, resp, design


def check_design(G, g, path):
    """Refuse anything that is not the complete factorial this grid claims to be."""
    shape = g['shape']
    n_cells = int(np.prod(shape))
    if G.shape[1] != len(shape):
        raise FormatError('the design has %d factors, this grid has %d'
                          % (G.shape[1], len(shape)))
    if G.shape[0] != n_cells:
        raise FormatError('%d architectures for the %s grid of %d cells: this is not the '
                          'complete factorial the decomposition needs'
                          % (G.shape[0], 'x'.join(str(q) for q in shape), n_cells))
    lin = np.ravel_multi_index(G.T, shape)
    uniq = np.unique(lin)
    if len(uniq) != n_cells:
        raise FormatError('%d distinct cells out of %d: the design misses %d of them and '
                          'covers others more than once, so it is not the complete factorial '
                          'the decomposition needs'
                          % (len(uniq), n_cells, n_cells - len(uniq)))
    for i, q in enumerate(shape):
        if len(np.unique(G[:, i])) != q:
            raise FormatError('factor %d takes %d of its %d levels'
                              % (i, len(np.unique(G[:, i])), q))


def run_grid(g, path, used_env):
    """Decompose every response of one grid. Returns the result block for the output."""
    G, resp, design = read_table(path, g)
    check_design(G, g, path)
    shape = g['shape']
    null = anova.null_spectrum(shape)
    block = {'benchmark': g['name'], 'source': g['source'],
             'shape': list(shape), 'n_factors': len(shape),
             'n_arch': int(G.shape[0]), 'note': g['note'],
             'levels': list(g['levels']),
             'provenance': {'path': shown(path, used_env),
                            'sha256': sha256(path),
                            'from_environment': g['env'] if used_env else None,
                            'bytes': int(os.path.getsize(path)),
                            'design_column': design},
             'null': {'per_order': null, 'cumulative': anova.cumulative(null)},
             'expected_responses': list(g['expected']),
             'responses': {}}
    for name in sorted(resp):
        Y = anova.to_grid(G, resp[name], shape)
        anova.check_consistency(Y)
        sp = anova.spectrum(Y)
        cum = anova.cumulative(sp)
        block['responses'][name] = {
            'per_order': sp, 'cumulative': cum,
            'null_ratio': [sp[i] / null[i] for i in range(len(sp))],
            'compression': compression(resp[name])}
    missing = [c for c in g['expected'] if c not in block['responses']]
    block['missing_expected_responses'] = missing
    return block


def tab_extra(blocks):
    """tables/tab_extra.tex, in the style of tables/tab_bench.tex."""
    letters = 'abcdefgh'
    what = ('a further complete factorial' if len(blocks) == 1
            else 'the %d further complete factorials' % len(blocks))
    pretty = {'acc_cifar10': 'CIFAR-10', 'acc_cifar100': 'CIFAR-100',
              'acc_ImageNet16-120': 'ImageNet16-120'}
    lab = lambda n: pretty.get(n, n.replace('_', r'\_'))
    L = [r'\begin{table}[!htb]', r'\centering',
         r'\caption{The exactness condition on ' + what + r', '
         r'recomputed by \texttt{analysis/s63\_extra\_grids.py} from a local copy of the '
         r'benchmark, which is not redistributed with this archive '
         r'(\texttt{data/external/PROVENANCE\_EXTRA.md}). Shares of accuracy variance by '
         r'interaction order, in per cent, each against the analytic white-noise null of its '
         r'own grid shape. Nothing is fitted: the design is verified to be the complete '
         r'factorial claimed, covered exactly once, and decomposed by the same Moebius '
         r'inversion as Tables~I and~II.}',
         r'\label{tab:extra}', r'\footnotesize',
         r'\setlength{\tabcolsep}{3.0pt}',
         r'\begin{tabular}{l rrrrr}', r'\toprule']
    for i, b in enumerate(blocks):
        if i:
            L.append(r'\addlinespace[2pt]\midrule')
        L.append(r'\multicolumn{6}{l}{\emph{(%s) %s}} \\' % (letters[i], b['tex']))
        L.append(r'order & 1 & 2 & 3 & $\le 2$ & $\le 3$ \\')
        L.append(r'\midrule')
        for name in b['order']:
            r = b['responses'][name]
            sp, cum = r['per_order'], r['cumulative']
            L.append(r'%s & %.1f & %.1f & %.1f & \textbf{%.1f} & \textbf{%.1f} \\'
                     % (lab(name), sp[0], sp[1], sp[2], cum[1], cum[2]))
        nul = b['null']
        L.append(r'white-noise null & %.1f & %.1f & %.1f & %.1f & %.1f \\'
                 % (nul['per_order'][0], nul['per_order'][1], nul['per_order'][2],
                    nul['cumulative'][1], nul['cumulative'][2]))
        #: skewness and the top-decile spread of every response, three to a line so that seven
        #: tasks do not become seven rows
        items = ['%s $%.2f$/%.1f\\%%' % (lab(n),
                                     b['responses'][n]['compression']['skewness'],
                                     100 * b['responses'][n]['compression']['top_decile_spread'])
                 for n in b['order']]
        for j in range(0, len(items), 3):
            lead = 'skewness/top-decile spread: ' if j == 0 else ''
            L.append(r'\multicolumn{6}{l}{\quad\footnotesize %s%s} \\'
                     % (lead, ', '.join(items[j:j + 3])))
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(L)


def main():
    out = {'grids': {}, 'skipped': [], 'refused': []}
    blocks, failed = [], False

    for g in GRIDS:
        path, used_env = resolve(g)
        disp = shown(path, used_env)
        print('%s' % g['name'])
        print('  %s: %s' % ('%s (from %s)' % (disp, g['env']) if used_env else disp,
                            'present' if os.path.exists(path) else 'not found'))
        if not os.path.exists(path):
            print('  skipped -- the table is not redistributed by this archive.')
            for line in g['fetch']:
                print('    ' + line)
            print('    The expected format is printed by this script when a file is present '
                  'but unreadable,')
            print('    and is documented in data/external/PROVENANCE_EXTRA.md.')
            out['skipped'].append({'grid': g['key'], 'expected_at': disp,
                                   'environment_override': g['env']})
            print()
            continue
        try:
            block = run_grid(g, path, used_env)
        except FormatError as exc:
            failed = True
            print('  refused -- the file is there but is not the export this script expects.')
            print('    %s' % exc)
            print()
            print(EXPECTED_FORMAT)
            print()
            out['refused'].append({'grid': g['key'], 'path': disp, 'reason': str(exc)})
            continue
        block['tex'] = g['tex']
        block['order'] = ([c for c in g['expected'] if c in block['responses']]
                          + [c for c in sorted(block['responses'])
                             if c not in g['expected']])
        out['grids'][g['key']] = block
        blocks.append(block)
        print('  complete %s factorial, %d architectures, %d responses, sha256 %s'
              % ('x'.join(str(q) for q in block['shape']), block['n_arch'],
                 len(block['responses']), block['provenance']['sha256'][:16]))
        print()

    if not blocks:
        print('No grid was computed, so nothing was written: neither '
              'analysis/results/extra_grids.json')
        print('nor tables/tab_extra.tex exists for a run with no table to read. Supply either')
        print('table as described above and rerun; every number in both outputs then comes '
              'from it.')
        return 1 if failed else 0

    # ---------------- report ----------------
    for b in blocks:
        d = b['n_factors']
        print('%s -- complete %s factorial, %d architectures'
              % (b['benchmark'], 'x'.join(str(q) for q in b['shape']), b['n_arch']))
        print('  %s' % b['note'])
        print('  %-22s' % 'order' + ''.join('%7d' % k for k in range(1, d + 1)))

        def row(lab, v, f='%7.2f'):
            print(('  %-22s' % lab) + ''.join(f % x for x in v))

        for name in b['order']:
            row(name, b['responses'][name]['per_order'])
        row('white-noise null', b['null']['per_order'])
        print()
        print('  %-22s %9s %9s %9s %9s' % ('response', 'order<=2', 'order<=3', 'skewness',
                                           'top-dec.'))
        for name in b['order']:
            r = b['responses'][name]
            print('  %-22s %8.1f%% %8.1f%% %9.2f %9.3f'
                  % (name, r['cumulative'][1], r['cumulative'][2],
                     r['compression']['skewness'], r['compression']['top_decile_spread']))
        print('  %-22s %8.1f%% %8.1f%%'
              % ('white-noise null', b['null']['cumulative'][1], b['null']['cumulative'][2]))
        if b['missing_expected_responses']:
            print('  note: the export does not carry %s'
                  % ', '.join(b['missing_expected_responses']))
        print()

    with open(os.path.join(RES, 'extra_grids.json'), 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    with open(os.path.join(TBL, 'tab_extra.tex'), 'w') as fh:
        fh.write(tab_extra(blocks))
    print('wrote analysis/results/extra_grids.json')
    print('wrote tables/tab_extra.tex (%d of %d grids)' % (len(blocks), len(GRIDS)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
