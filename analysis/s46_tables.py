"""Compose the one merged table the proxy comparison needs in the ten-page manuscript.

`s43_zc_proxies.py` and `s44_zc_spaces.py` each write their own table, and both are in the
supplement. The main paper has room for one, so this script merges them: the free ranker
against the zero-cost proxy family, in rank fidelity and in regret, across the three public
spaces the proxy release covers. (The paper measures the ranker itself in a fourth, our own,
where no published proxy values exist -- which is why the manuscript says "three spaces"
wherever the claim is about the proxy comparison.) The label-free block is the comparison that
matters: it is the only one in which both rankers are free.

  tab_zc_main.tex   zc_nb201.json + zc_spaces.json + encoding.json   -> Table V
"""
import json, os
import common

DS = ['cifar10', 'cifar100', 'ImageNet16-120']
SHORT = {'cifar10': 'C10', 'cifar100': 'C100', 'ImageNet16-120': 'IN16'}


def J(n):
    return json.load(open(os.path.join(common.RES, n + '.json')))


def w(name, lines):
    p = os.path.join(common.HERE, '..', 'tables', name)
    open(p, 'w').write('\n'.join(lines) + '\n')
    print('wrote', p)


def cell(e, what):
    if e is None:
        return '---'
    if what == 'rho':
        return '%.3f' % e['rho']
    return '%.2f' % e['regret']['10']


def main():
    z = J('zc_nb201')
    s = J('zc_spaces')
    enc = J('encoding')
    pd = z['per_dataset']
    L = []
    a = L.append
    a(r'\begin{table}[!htb]')
    a(r'\centering')
    a(r'\caption{The free ranker against the zero-cost proxy family, in the three discrete '
      r'spaces that release covers. All 13 proxy scores are taken unchanged from '
      r'NAS-Bench-Suite-Zero~\cite{krishnakumar2022nasbenchsuitezero}; every ranker in a '
      r'column is scored on one held-out pool over the same 20 splits. $r_{10}$ is the points '
      r'given up by training the ten best-ranked candidates instead of the whole pool. The '
      r'middle block is the only comparison in which \emph{both} rankers are free. '
      r'NAS-Bench-101 is the failure case; a negative held-out $R^{2}$ identifies it for '
      r'nothing. Per-proxy detail: Supplementary Tables~S8 and~S9.}')
    a(r'\label{tab:zc}')
    a(r'\footnotesize')
    a(r'\setlength{\tabcolsep}{2.6pt}')
    a(r'\resizebox{\columnwidth}{!}{%')
    a(r'\begin{tabular}{l rr rr rr rr rr}')
    a(r'\toprule')
    a(r'& \multicolumn{6}{c}{NAS-Bench-201} & \multicolumn{2}{c}{NB-301}'
      r' & \multicolumn{2}{c}{NB-101} \\')
    a(r'\cmidrule(lr){2-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}')
    a(r'& \multicolumn{2}{c}{C10} & \multicolumn{2}{c}{C100} & \multicolumn{2}{c}{IN16}'
      r' & \multicolumn{2}{c}{C10} & \multicolumn{2}{c}{C10} \\')
    a(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'
      r'\cmidrule(lr){8-9}\cmidrule(lr){10-11}')
    a(r'Ranker & $\rho$ & $r_{10}$ & $\rho$ & $r_{10}$ & $\rho$ & $r_{10}$'
      r' & $\rho$ & $r_{10}$ & $\rho$ & $r_{10}$ \\')
    a(r'\midrule')

    def row(label, keys, bold=False):
        """keys = (nb201 key, nb301 key, nb101 key); None skips a column."""
        k201, k301, k101 = keys
        cs = []
        for d in DS:
            e = pd[d].get(k201) if k201 else None
            cs += [cell(e, 'rho'), cell(e, 'r10')]
        e = s['nb301']['rankers'].get(k301) if k301 and 'nb301' in s else None
        cs += [cell(e, 'rho'), cell(e, 'r10')]
        e = s['nb101']['rankers'].get(k101) if k101 and 'nb101' in s else None
        cs += [cell(e, 'rho'), cell(e, 'r10')]
        lab = (r'\textbf{%s}' % label) if bold else label
        a(r'\quad ' + lab + ' & ' + ' & '.join(cs) + r' \\')

    a(r'\multicolumn{11}{l}{\emph{Needs labels of the target dataset}} \\')
    row('Genotype surrogate, 100 labels',
        ('surrogate|100', 'surrogate|100', 'surrogate|100'))
    row('Genotype surrogate, 1{,}000 labels',
        ('surrogate|1000', 'surrogate|1000', 'surrogate|1000'))
    a(r'\addlinespace[1.5pt]')
    a(r'\multicolumn{11}{l}{\emph{Needs no label of the target: the free rankers}} \\')
    for m, nice in [('synflow', 'SynFlow'), ('nwot', 'NWOT'), ('zen', 'Zen'),
                    ('params', 'Params')]:
        row('%s (zero-cost proxy)' % nice,
            ('proxy|' + m, 'proxy|' + m, 'proxy|' + m))
    row('Rank-mean of all 13 proxies', ('fused|all13', 'fused|all13', 'fused|all13'))
    cs = []
    for d in DS:
        cs += [cell(pd[d]['proxy_picked|source'], 'rho'),
               cell(pd[d]['proxy_picked|source'], 'r10')]
    cs += ['---'] * 4
    a(r'\quad Proxy picked on the source labels & ' + ' & '.join(cs) + r' \\')
    # the transferred surrogate exists only where the space has more than one dataset
    cs = []
    for d in DS:
        src = z['transfer_source_rule']['source'][d]
        cs += [cell(pd[d]['transfer|' + src], 'rho'),
               cell(pd[d]['transfer|' + src], 'r10')]
    cs += ['---'] * 4
    a(r'\quad \textbf{Surrogate transferred in} & ' + ' & '.join(cs) + r' \\')
    cs = []
    for d in DS:
        cs += [cell(pd[d]['fused|transfer+picked'], 'rho'),
               cell(pd[d]['fused|transfer+picked'], 'r10')]
    cs += ['---'] * 4
    a(r'\quad \textbf{\quad $+$ a proxy, 2:1} & ' + ' & '.join(cs) + r' \\')
    cs = []
    for d in DS:
        cs += ['---', '%.2f' % z['label_free'][d]['oracle_regret10']]
    cs += ['---'] * 4
    a(r'\quad \emph{oracle proxy choice (ceiling)} & ' + ' & '.join(cs) + r' \\')
    a(r'\addlinespace[1.5pt]')
    a(r'\multicolumn{11}{l}{\emph{The failure case, and the repair}} \\')
    e = enc['nb101']['rankers']['path|1000']
    a(r'\quad Path encoding, 1{,}000 labels & ' + ' & '.join(['---'] * 8)
      + ' & %.3f & %.2f' % (e['rho'], e['regret']['10']) + r' \\')
    a(r'\quad Held-out $R^{2}$, genotype ridge & '
      + ' & '.join(r'\multicolumn{2}{c}{$%+.2f$}' % enc['diagnostic'][k]['test']
                   for k in ['nb201_cifar10_genotype', 'nb201_cifar100_genotype',
                             'nb201_ImageNet16-120_genotype', 'nb301_genotype',
                             'nb101_genotype'])
      + r' \\')
    a(r'\addlinespace[1.5pt]\midrule')
    cs = []
    for d in DS:
        cs += ['%.2f' % abs(pd[d]['random|']['rho']),
               '%.2f' % pd[d]['random|']['regret']['10']]
    for nm in ['nb301', 'nb101']:
        e = s['nm'] if False else (s[nm]['rankers']['random|'] if nm in s else None)
        cs += [('%.2f' % abs(e['rho'])) if e else '---',
               ('%.2f' % e['regret']['10']) if e else '---']
    a(r'\quad Random shortlist & ' + ' & '.join(cs) + r' \\')
    a(r'\bottomrule')
    a(r'\end{tabular}}')
    a(r'\end{table}')
    w('tab_zc_main.tex', L)


if __name__ == '__main__':
    main()
