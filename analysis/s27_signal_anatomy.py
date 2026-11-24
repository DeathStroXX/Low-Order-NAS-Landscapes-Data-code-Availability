"""Steps 3-5 -- the anatomy of the ranking signal: what it needs, where it works, what it misses.

(a) Identifiability. A pairwise model over 13 genes has one cell per ordered gene-pair value
    combination -- 1,514 of them. The released runs cover 0.012% of the 42.5-million-point grid
    and yet every one of those cells, which is why a few hundred labels suffice. The label
    budget a practitioner needs scales with sum |g_a||g_b|, not with the size of the space.

(b) Regime. A global Spearman rho flatters every surrogate, because most of the ranking work is
    separating good architectures from hopeless ones. Selection late in a run only ever compares
    architectures that are already good, so fidelity is re-measured inside the top decile.

(c) Residual information. After removing everything the genotype explains, how much does a
    measured epoch still know, and how much does the LLM predictor? That decides which signals
    belong in a two-stage ranker and in what order.

Writes results/signal_anatomy.json, tables/tab_regime.tex, tables/tab_residual.tex.
"""
import itertools, json
import numpy as np
import common, surrogate as sg
from scipy.stats import spearmanr, rankdata

FOLDS = 5
QUANTILES = [(0.0, 'all'), (0.5, 'top 50\\%'), (0.9, 'top 10\\%'), (0.95, 'top 5\\%')]


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def cv_pred(P, y, idx):
    """Out-of-fold prediction of `y` from the genotype, same folds throughout."""
    gf = sg.GramFitter(P, y)
    out = np.zeros(len(y))
    for f in range(FOLDS):
        te = idx[f::FOLDS]
        keep = np.ones(len(y), bool)
        keep[te] = False
        out[te] = gf.fit_mask(keep)(te)
    return out


def main():
    out = {}
    for ds in common.DATASETS:
        rows = [r for r in common.gene_table()
                if r['dataset'] == ds and r['acc3'] is not None]
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        a1 = np.array([r['acc1'] if r['acc1'] is not None else np.nan for r in rows])
        pr = np.array([r['pred'] if r['pred'] is not None else np.nan for r in rows])
        n = len(y)
        idx = np.random.default_rng(0).permutation(n)
        e = {}

        # ---- (a) identifiability -------------------------------------------------
        cells = sum(int(sg.CARD[a] * sg.CARD[b])
                    for a, b in itertools.combinations(range(sg.NG), 2))
        seen = sum(len({(int(p[a]), int(p[b])) for p in P})
                   for a, b in itertools.combinations(range(sg.NG), 2))
        uniq = len({tuple(int(v) for v in p) for p in P})
        # additive-only control, same folds
        A = sg.design(P)[:, :sg.NMAIN].astype(np.float64)
        pm_ = np.zeros(n)
        for f in range(FOLDS):
            te = idx[f::FOLDS]
            tr = np.setdiff1d(idx, te)
            G = A[tr].T @ A[tr]
            G[np.diag_indices_from(G)] += 10.0
            b = y[tr].mean()
            w = np.linalg.solve(G, A[tr].T @ (y[tr] - b))
            pm_[te] = A[te] @ w + b
        p_sur = cv_pred(P, y, idx)
        r2 = lambda p: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        e['identifiability'] = {
            'n': n, 'unique_genotypes': uniq, 'grid': sg.GRID_SIZE,
            'grid_fraction': uniq / sg.GRID_SIZE,
            'pair_cells': cells, 'pair_cells_seen': seen,
            'labels_per_cell': n / cells,
            'additive': {'rho': rho(pm_, y), 'r2': r2(pm_)},
            'pairwise': {'rho': rho(p_sur, y), 'r2': r2(p_sur)}}

        # ---- (b) regime ----------------------------------------------------------
        ok1 = ~np.isnan(a1)
        reg = {}
        for q, name in QUANTILES:
            m = y >= np.quantile(y, q)
            m1 = m & ok1
            reg[name] = {'n': int(m.sum()),
                         'surrogate': rho(p_sur[m], y[m]),
                         'epoch1': rho(a1[m1], y[m1])}
        e['regime'] = reg

        # ---- (c) residual information -------------------------------------------
        okp = ok1 & ~np.isnan(pr)
        Pk, yk, ak, pk = P[okp], y[okp], a1[okp], pr[okp]
        ik = np.random.default_rng(0).permutation(len(yk))
        s_k = cv_pred(Pk, yk, ik)
        res_y = yk - s_k
        res_e1 = ak - cv_pred(Pk, ak, ik)
        res_pr = pk - cv_pred(Pk, pk, ik)
        rs, re_, rp = rankdata(s_k), rankdata(ak), rankdata(pk)
        e['residual'] = {
            'n': int(okp.sum()),
            'epoch1_vs_truth': rho(res_e1, res_y),
            'predictor_vs_truth': rho(res_pr, res_y),
            'fusion': {'surrogate': rho(s_k, yk), 'epoch1': rho(ak, yk),
                       'predictor': rho(pk, yk),
                       'surrogate+epoch1': rho(2 * rs + re_, yk),
                       'surrogate+epoch1+predictor': rho(2 * rs + 2 * re_ + rp, yk)}}
        out[ds] = e
    json.dump(out, open(f'{common.RES}/signal_anatomy.json', 'w'), indent=1)

    # ---- table: one anatomy of the ranking signal ---------------------------------
    S = json.load(open(f'{common.RES}/surrogate.json'))
    c10, c100 = out['cifar10'], out['cifar100']
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Anatomy of the ranking signal. Top: rank fidelity of every signal the',
         r'pipeline had available. Middle: the same fidelity restricted to the architectures',
         r'selection actually compares late in a run. Bottom: what each signal knows that the',
         r'genotype does not, measured as the correlation between its residual and the',
         r'truth\'s, both residualized on the same out-of-fold genotype model.}',
         r'\label{tab:signal}', r'\footnotesize', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lcc}', r'\toprule',
         r'& CIFAR-10 & CIFAR-100 \\', r'\midrule',
         r'\multicolumn{3}{l}{\emph{Rank fidelity against the 3-epoch reference}} \\',
         rf'\quad Genotype surrogate (free) & \textbf{{{S["cifar10"]["rho"]["surrogate"]:.3f}}} & '
         rf'\textbf{{{S["cifar100"]["rho"]["surrogate"]:.3f}}} \\',
         rf'\quad One measured training epoch & {S["cifar10"]["rho"]["epoch1"]:.3f} & '
         rf'{S["cifar100"]["rho"]["epoch1"]:.3f} \\',
         rf'\quad LoRA-tuned 7B predictor (the fitness) & '
         rf'{S["cifar10"]["rho"]["predictor_llm"]:.3f} & '
         rf'{S["cifar100"]["rho"]["predictor_llm"]:.3f} \\',
         rf'\quad Surrogate $+$ epoch, rank-fused 2:1 & '
         rf'{c10["residual"]["fusion"]["surrogate+epoch1"]:.3f} & '
         rf'{c100["residual"]["fusion"]["surrogate+epoch1"]:.3f} \\',
         rf'\quad \dots\ $+$ the predictor & '
         rf'{c10["residual"]["fusion"]["surrogate+epoch1+predictor"]:.3f} & '
         rf'{c100["residual"]["fusion"]["surrogate+epoch1+predictor"]:.3f} \\',
         r'\addlinespace[2pt]', r'\midrule',
         r'\multicolumn{3}{l}{\emph{Surrogate / one epoch, by selection regime}} \\']
    for _, name in QUANTILES:
        a, b = c10['regime'][name], c100['regime'][name]
        L.append(rf'\quad {name} ($n={a["n"]}$, ${b["n"]}$) & '
                 rf'{a["surrogate"]:.3f} / {a["epoch1"]:.3f} & '
                 rf'{b["surrogate"]:.3f} / {b["epoch1"]:.3f} \\')
    L += [r'\addlinespace[2pt]', r'\midrule',
          r'\multicolumn{3}{l}{\emph{Residual correlation with the truth, genotype removed}} \\',
          rf'\quad One measured training epoch & '
          rf'\textbf{{{c10["residual"]["epoch1_vs_truth"]:+.3f}}} & '
          rf'\textbf{{{c100["residual"]["epoch1_vs_truth"]:+.3f}}} \\',
          rf'\quad LoRA-tuned 7B predictor & {c10["residual"]["predictor_vs_truth"]:+.3f} & '
          rf'{c100["residual"]["predictor_vs_truth"]:+.3f} \\',
          r'\addlinespace[2pt]', r'\midrule',
          r'\multicolumn{3}{l}{\emph{Identifiability of the pairwise model}} \\',
          rf'\quad Gene-pair cells covered & '
          rf'{c10["identifiability"]["pair_cells_seen"]:,}/'
          rf'{c10["identifiability"]["pair_cells"]:,} & '
          rf'{c100["identifiability"]["pair_cells_seen"]:,}/'
          rf'{c100["identifiability"]["pair_cells"]:,} \\'.replace(',', '{,}'),
          rf'\quad \dots\ from this share of the grid & '
          rf'{100*c10["identifiability"]["grid_fraction"]:.4f}\% & '
          rf'{100*c100["identifiability"]["grid_fraction"]:.4f}\% \\',
          rf'\quad $R^2$, additive only $\to$ with gene pairs & '
          rf'{c10["identifiability"]["additive"]["r2"]:.3f} $\to$ '
          rf'{c10["identifiability"]["pairwise"]["r2"]:.3f} & '
          rf'{c100["identifiability"]["additive"]["r2"]:.3f} $\to$ '
          rf'{c100["identifiability"]["pairwise"]["r2"]:.3f} \\',
          r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_signal.tex', 'w').write('\n'.join(L) + '\n')

    for ds, e in out.items():
        i = e['identifiability']
        print(f'== {ds} ==')
        print(f'  identifiability: {i["unique_genotypes"]} genotypes = '
              f'{100*i["grid_fraction"]:.4f}% of the grid, covering '
              f'{i["pair_cells_seen"]}/{i["pair_cells"]} pair cells '
              f'({i["labels_per_cell"]:.1f} labels per cell); '
              f'additive R2={i["additive"]["r2"]:.3f} -> pairwise R2={i["pairwise"]["r2"]:.3f}')
        for _, name in QUANTILES:
            r_ = e['regime'][name]
            print(f'  regime {name:10s} n={r_["n"]:5d}  surrogate={r_["surrogate"]:.3f}  '
                  f'epoch1={r_["epoch1"]:.3f}')
        r_ = e['residual']
        print(f'  residual beyond genotype: epoch1={r_["epoch1_vs_truth"]:+.3f}  '
              f'predictor={r_["predictor_vs_truth"]:+.3f}  '
              f'fusion(s+e)={r_["fusion"]["surrogate+epoch1"]:.3f}')


if __name__ == '__main__':
    main()
