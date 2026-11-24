"""Refresh the two merged supplementary tables from analysis/results/.

tables/tab_sitl2.tex (Table S-sitl: model-in-the-loop search with the warm-start block below
it) and tables/tab_transfer2.tex (transfer in our own space above NAS-Bench-201) were merged by
hand from the outputs of s31_sitl.py, s29_warmstart.py, s25_transfer.py and s35_nb201.py, so a
rerun of those scripts used to leave the merged copies -- which are the ones the supplement
inlines -- stale. This script rewrites their numeric rows in place from the result files and
leaves the captions and the layout alone, so it can be run after every analysis rerun.

Run before s70_assemble.py:
    python3 s71_merged_tables.py && python3 s70_assemble.py
"""
import json, os, re

import common

A, B = 'cifar10', 'cifar100'
TAB = os.path.join(common.HERE, '..', 'tables')


def J(n):
    return json.load(open(os.path.join(common.RES, n + '.json')))


def replace_row(text, prefix, cells):
    """Replace the cells of the one tabular row that starts with `prefix`."""
    pat = re.compile(r'^(' + re.escape(prefix) + r')\s*&.*?\\\\\s*$', re.M)
    hits = pat.findall(text)
    if len(hits) != 1:
        raise ValueError('row %r found %d times' % (prefix, len(hits)))
    return pat.sub(lambda m: m.group(1) + ' & ' + ' & '.join(cells) + r' \\', text)


def sitl2():
    si, wm = J('sitl'), J('warmstart')
    path = os.path.join(TAB, 'tab_sitl2.tex')
    t = open(path).read()
    budgets = ['120', '260', '520']
    rows = [('GA, every individual trained', 'plain'),
            ('Model in the loop, from scratch', 'scratch'),
            (r'\quad $+$ random exploration', 'explore'),
            (r'\quad $+$ transferred prior', 'transfer')]
    for label, key in rows:
        t = replace_row(t, label, ['%.2f' % si[ds][b][key] for ds in (A, B) for b in budgets])
    diff = [r'\textbf{%+.2f}' % (si[ds][b]['transfer'] - si[ds][b]['plain'])
            for ds in (A, B) for b in budgets]
    t = replace_row(t, r'\textbf{Transferred prior $-$ GA}', diff)
    # the warm-start block: "cold / warm" accuracy at each selection-signal fidelity
    for label, key in [(r'\quad exact ($\rho=1$)', 'exact'), (r'\quad $\rho=0.93$', '0.93'),
                       (r'\quad $\rho=0.70$', '0.7'), (r'\quad $\rho=0.58$', '0.58'),
                       (r'\quad $\rho=0.43$', '0.43'), (r'\quad $\rho=0.33$', '0.33')]:
        cells = []
        for ds in (A, B):
            c = _warm(wm[ds], key)
            cell = '%.2f / %.2f' % c
            if key == wm[ds]['predictor_level']:      # the predictor's own fidelity
                cell = r'\textbf{%s}' % cell
            cells.append(r'\multicolumn{3}{c}{%s}' % cell)
        t = replace_row(t, label, cells)
    # the caption quotes where the seeded loop starts
    t = re.sub(r'it starts at \$\\rho=[0-9.]+\$', lambda m: r'it starts at $\rho=%.2f$'
               % si[A]['prior_rho'], t)
    open(path, 'w').write(t)
    print('wrote tab_sitl2.tex')


def _warm(w, key):
    """(cold, warm) accuracy of the as-run warm-start experiment at one signal fidelity."""
    r = w['asrun']
    if key == 'exact':
        return r['clean'], r['clean_warm']
    return r[key]['noisy'], r[key]['warm']


def transfer2():
    tr = J('transfer')
    path = os.path.join(TAB, 'tab_transfer2.tex')
    t = open(path).read()
    for label, src, tgt in [(r'\quad C10 & C100', 'cifar10', 'cifar100'),
                            (r'\quad C100 & C10', 'cifar100', 'cifar10')]:
        d = tr['dataset'][src + '->' + tgt]
        rho, own, ep = d['few_shot']['0']['transfer'], d['few_shot']['500']['scratch'], \
            d['epoch1_on_target']
        cell = r'\textbf{%.3f}' % rho if rho > own else '%.3f' % rho
        # only the FractalNet rows: they are the ones with an epoch column that is not ---
        pat = re.compile(r'^(' + re.escape(label) + r')\s*&[^\n]*?& [0-9.]+ \\\\\s*$', re.M)
        hits = pat.findall(t)
        if len(hits) != 1:
            raise ValueError('FractalNet row %r found %d times' % (label, len(hits)))
        t = pat.sub(lambda m: m.group(1) + ' & %s & %.3f & %.3f \\\\' % (cell, own, ep), t)
    open(path, 'w').write(t)
    print('wrote tab_transfer2.tex')


if __name__ == '__main__':
    sitl2()
    transfer2()
