"""Step 2 -- the decision metric: how many architectures must actually be trained?

A rank correlation answers a question nobody asks. The question a practitioner asks is: if I
let a free model pick a shortlist and I train only that shortlist, how much accuracy do I give
up against training everything? That is *regret at k*, and unlike $\\rho$ it does not have to
degrade as the comparison narrows, because a shortlist only has to *contain* a good
architecture, not order it correctly.

This script measures regret at k for every signal the pipeline had, on the released
per-architecture record, with the surrogate always fitted on a disjoint half so nothing is
circular. It also prices the whole ladder of cheap signals in one table: what each rung costs
in training epochs, the fidelity it reaches, and the regret it leaves.

Writes results/regret.json, ../tables/tab_regret.tex, ../figures/fig_regret.pdf.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import common
import surrogate as sg

KS = [1, 2, 5, 10, 20, 50, 100]
SPLITS = 40
ALPHA = 10.0


def rk(v):
    """Ranks in [0, 1], ties averaged."""
    v = np.asarray(v, float)
    o = np.argsort(np.argsort(v))
    return o / max(len(v) - 1, 1)


def regret(sig, y, k):
    return float(y.max() - y[np.argsort(-sig)[:k]].max())


def main():
    table = [r for r in common.gene_table()
             if r['acc1'] is not None and r['acc3'] is not None]
    out = {'splits': SPLITS, 'ks': KS}

    for ds in common.DATASETS:
        rows = [r for r in table if r['dataset'] == ds]
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        a1 = np.array([r['acc1'] for r in rows])
        pr = np.array([r['pred'] if r['pred'] is not None else np.nan for r in rows])
        n = len(y)

        # the surrogate transferred from the *other* dataset: no label of this one
        orows = [r for r in table if r['dataset'] != ds]
        oP = np.array([sg.codes_of(r) for r in orows])
        oy = np.array([r['acc3'] for r in orows])
        xmodel = sg.PairwiseRidge(ALPHA).fit(oP, oy)

        acc = {s: {str(k): [] for k in KS} for s in
               ['surrogate', 'epoch1', 'predictor', 'fused', 'transferred', 'random']}
        rhos = {s: [] for s in acc}
        rho_top = {s: [] for s in acc}

        for seed in range(SPLITS):
            g = np.random.default_rng(seed)
            idx = g.permutation(n)
            tr, te = idx[:n // 2], idx[n // 2:]
            m = sg.PairwiseRidge(ALPHA).fit(P[tr], y[tr])
            yt = y[te]
            sig = {'surrogate': m.predict(P[te]),
                   'epoch1': a1[te],
                   'predictor': np.where(np.isnan(pr[te]), -1e9, pr[te]),
                   'transferred': xmodel.predict(P[te]),
                   'random': g.standard_normal(len(te))}
            sig['fused'] = (2 * rk(sig['surrogate']) + rk(sig['epoch1'])) / 3
            thr = np.quantile(yt, 0.9)
            hi = yt >= thr
            for s, v in sig.items():
                for k in KS:
                    acc[s][str(k)].append(regret(v, yt, k))
                # a signal that is missing for some architectures (the LLM predictor runs
                # only after training, and a few logs lack it) is ranked where it exists;
                # for regret, a missing value simply never reaches a shortlist.
                ok = np.isfinite(pr[te]) if s == 'predictor' else np.ones(len(te), bool)
                rhos[s].append(float(spearmanr(v[ok], yt[ok]).correlation))
                h2 = hi & ok
                rho_top[s].append(float(spearmanr(v[h2], yt[h2]).correlation)
                                  if h2.sum() > 10 else float('nan'))

        out[ds] = {
            'n': n,
            'pool_best': float(y.max()),
            'pool_mean': float(y.mean()),
            'regret': {s: {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                               'p90': float(np.quantile(v, 0.9))}
                           for k, v in d.items()} for s, d in acc.items()},
            'rho': {s: float(np.mean(v)) for s, v in rhos.items()},
            'rho_top10': {s: float(np.mean(v)) for s, v in rho_top.items()},
        }

        # ---------------------------------------------------------------- the ladder
        # Cost is counted in training epochs per candidate screened. Rungs are cumulative.
        lad = []
        for name, cost, key in [
                ('Genotype surrogate, transferred', 0.0, 'transferred'),
                ('Genotype surrogate, fitted here', 0.0, 'surrogate'),
                ('One measured epoch', 1.0, 'epoch1'),
                ('Surrogate $+$ epoch, fused 2:1', 1.0, 'fused'),
                ('LoRA-tuned 7B predictor', 1.0, 'predictor')]:
            lad.append({'rung': name, 'epochs_per_candidate': cost,
                        'rho': out[ds]['rho'][key],
                        'rho_top10': out[ds]['rho_top10'][key],
                        'regret20': out[ds]['regret'][key]['20']['mean'],
                        'regret5': out[ds]['regret'][key]['5']['mean']})
        out[ds]['ladder'] = lad

        # k* : the smallest shortlist whose regret is under a tenth of a point
        for s in acc:
            ks = [k for k in KS if out[ds]['regret'][s][str(k)]['mean'] <= 0.1]
            out[ds].setdefault('k_star', {})[s] = (min(ks) if ks else None)

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'regret.json'), 'w'), indent=1)
    emit_table(out)
    emit_figure(out)
    for ds in common.DATASETS:
        print(ds, 'k* surrogate =', out[ds]['k_star']['surrogate'],
              '| regret@20', round(out[ds]['regret']['surrogate']['20']['mean'], 3),
              '| rho_top10', round(out[ds]['rho_top10']['surrogate'], 3),
              '| transferred regret@20',
              round(out[ds]['regret']['transferred']['20']['mean'], 3))


def emit_table(o):
    A, B = 'cifar10', 'cifar100'
    L = []
    a = L.append
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{What a cheap signal is worth, in the unit the decision is made in. '
      r'\emph{Regret at $k$} is the accuracy lost by training only the $k$ architectures a '
      r'signal ranks highest, against training the whole pool; $\rho_{10}$ is rank fidelity '
      r'inside the top decile. The surrogate is always fitted on a disjoint half of the '
      r'record; \emph{transferred} uses no label of the target dataset at all. '
      r'40 random splits.}')
    a(r'\label{tab:regret}')
    a(r'\footnotesize')
    a(r'\resizebox{\columnwidth}{!}{%')
    a(r'\begin{tabular}{lcccccc}')
    a(r'\toprule')
    a(r'& \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{CIFAR-100} \\')
    a(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    a(r'Signal (epochs/candidate) & $\rho_{10}$ & $k{=}5$ & $k{=}20$'
      r' & $\rho_{10}$ & $k{=}5$ & $k{=}20$ \\')
    a(r'\midrule')
    for i, r10 in enumerate(o[A]['ladder']):
        r100 = o[B]['ladder'][i]
        a('%s (%g) & %.3f & %.2f & %.2f & %.3f & %.2f & %.2f \\\\' % (
            r10['rung'], r10['epochs_per_candidate'],
            r10['rho_top10'], r10['regret5'], r10['regret20'],
            r100['rho_top10'], r100['regret5'], r100['regret20']))
    a(r'\midrule')
    a('Random shortlist (0) & --- & %.2f & %.2f & --- & %.2f & %.2f \\\\' % (
        o[A]['regret']['random']['5']['mean'], o[A]['regret']['random']['20']['mean'],
        o[B]['regret']['random']['5']['mean'], o[B]['regret']['random']['20']['mean']))
    a(r'\bottomrule')
    a(r'\end{tabular}}')
    a(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_regret.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def emit_figure(o):
    """Single-column: regret against the number of architectures actually trained."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    STY = [('surrogate', '#1b6ca8', 'Genotype surrogate (free)'),
           ('transferred', '#2e7d4f', 'Transferred, no target label'),
           ('epoch1', '#c7522a', 'One measured epoch'),
           ('predictor', '#8d6ca8', 'LoRA-tuned 7B predictor'),
           ('random', '#8a8a8a', 'Random shortlist')]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for s, c, lab in STY:
        for ds, ls, mk in [('cifar10', '-', 'o'), ('cifar100', '--', 's')]:
            v = [o[ds]['regret'][s][str(k)]['mean'] for k in o['ks']]
            ax.plot(o['ks'], v, ls, marker=mk, color=c, lw=1.4, ms=3.0,
                    label=lab if ds == 'cifar10' else None)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.05)
    ax.axhline(0.1, color='k', ls=':', lw=.8)
    ax.set_xlabel('architectures actually trained ($k$)', fontsize=8)
    ax.set_ylabel('regret (accuracy points)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25, lw=.5)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color='k', ls='-', lw=1.2),
          Line2D([], [], color='k', ls='--', lw=1.2)]
    l += ['CIFAR-10', 'CIFAR-100']
    ax.legend(h, l, fontsize=5.6, frameon=False, loc='lower left', ncol=1,
              handlelength=1.6, labelspacing=.25)
    ax.text(0.98, 0.93, 'a tenth of a point', fontsize=5.6, ha='right',
            transform=ax.transAxes)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_regret.' + ext),
                    dpi=220, bbox_inches='tight')
    print('wrote figures/fig_regret.pdf')


if __name__ == '__main__':
    main()
