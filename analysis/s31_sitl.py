"""Surrogate-in-the-loop search with a surrogate that is really refitted.

Driving the loop with the benchmark's own value corrupted to the surrogate's measured rank
fidelity is not the same thing: it grants the loop a surrogate that is unbiased everywhere,
whereas a real one is fitted on exactly the architectures the loop chose to measure, which are
not a random sample, and the two reach opposite conclusions. This script therefore reports only
loops whose model is genuinely fitted on what has been measured.

Four variants, all at an identical number of *measured* architectures:

  plain      the GA as the pipeline runs it -- every individual trained
  scratch    model-in-the-loop, ridge refitted on the measured set, started from random labels
  explore    as scratch, but a share of each generation's measurements is drawn at random
  transfer   as explore, but the model starts as the ridge fitted on the *other dataset*
             (Section IV): a prior that costs no label on this dataset at all

Writes results/sitl.json, tables/tab_sitl.tex.
"""
import json
import numpy as np
import common, fractalbench as fb, policies, surrogate as sg
from surrogate import CARD, NG
from scipy.stats import spearmanr

BUDGETS = [120, 260, 520]
R = 40
POP, M, K, PM, ELITE = 40, 4, 10, 0.08, 2
WARM = 24


class SmallRidge:
    """The same pairwise ridge, solved in the dual.

    With n labels and p = 1,572 columns and n << p, solving the n x n system
    (A A^T + alpha I) c = y - b and taking w = A^T c is several times cheaper than the normal
    equations, which is what makes a refit inside every search loop affordable.
    """

    def __init__(self, alpha=10.0):
        self.alpha, self.w, self.b = alpha, None, 0.0

    def fit(self, P, y):
        A = sg.design(P).astype(np.float64)
        y = np.asarray(y, float)
        self.b = float(y.mean())
        G = A @ A.T
        G[np.diag_indices_from(G)] += self.alpha
        self.w = A.T @ np.linalg.solve(G, y - self.b)
        return self

    def predict(self, P):
        return sg.design(P) @ self.w + self.b


def alpha_for(n):
    """A pairwise ridge has 1,572 columns; with tens of labels it must be heavily penalised."""
    return float(np.clip(3000.0 / max(n, 1), 10.0, 600.0))


def prior_model(other_ds):
    rows = [r for r in common.gene_table()
            if r['dataset'] == other_ds and r['acc3'] is not None]
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows])
    return sg.PairwiseRidge(10.0).fit(P, y)


def loop(bench, budget, mode, prior=None, seed=3, n_explore=1, probe=None):
    """One model-in-the-loop search per replicate. Only measured architectures cost budget."""
    rng = np.random.default_rng(seed)
    best = np.full(R, -np.inf)
    fid = []
    for r in range(R):
        warm = 0 if mode == 'transfer' else WARM
        Xs = rng.integers(0, CARD, (max(warm, 1), NG)) if warm else np.zeros((0, NG), int)
        ys = bench.score(Xs) if warm else np.zeros(0)
        used = warm
        if warm:
            best[r] = ys.max()

        def refit():
            if prior is None:
                if len(ys) < 8:
                    return lambda Q: rng.random(len(Q))
                return SmallRidge(alpha_for(len(ys))).fit(Xs, ys).predict
            if len(ys) < 8:
                return prior.predict
            res = ys - prior.predict(Xs)
            corr = SmallRidge(alpha_for(len(ys)) * 3).fit(Xs, res)
            return lambda Q: prior.predict(Q) + corr.predict(Q)

        pred = refit()
        P = rng.integers(0, CARD, (POP, NG))
        g = 0
        while used < budget:
            f = pred(P)
            take = min(M, budget - used)
            ex = min(n_explore, take) if mode in ('explore', 'transfer') else 0
            top = np.argsort(-f)[:take - ex]
            pick = np.concatenate([top, rng.choice(POP, ex, replace=False)]) if ex else top
            t = bench.score(P[pick])
            Xs = np.concatenate([Xs, P[pick]])
            ys = np.concatenate([ys, t])
            used += len(pick)
            best[r] = max(best[r], t.max())
            if g % 5 == 0:
                pred = refit()
            o = np.argsort(-f)
            new = np.empty_like(P)
            new[:ELITE] = P[o[:ELITE]]
            n = POP - ELITE
            c = rng.integers(0, POP, (n, K))
            A = P[c[np.arange(n), f[c].argmax(1)]]
            c2 = rng.integers(0, POP, (n, K))
            B = P[c2[np.arange(n), f[c2].argmax(1)]]
            pt = rng.integers(1, NG, (n, 1))
            ch = np.where(np.arange(NG)[None, :] < pt, A, B)
            mm = rng.random((n, NG)) < PM
            sh = rng.integers(1, np.broadcast_to(CARD, (n, NG)))
            new[ELITE:] = np.where(mm, (ch + sh) % CARD, ch)
            P = new
            g += 1
        if probe is not None and r < 20:
            fid.append(float(spearmanr(pred(probe[0]), probe[1]).statistic))
    return {'best': float(best.mean()), 'se': float(best.std() / np.sqrt(R)),
            'fidelity': float(np.mean(fid)) if fid else None}


def main():
    out = {}
    for ds in common.DATASETS:
        bench = fb.Bench(ds)
        other = 'cifar100' if ds == 'cifar10' else 'cifar10'
        pri = prior_model(other)
        rng = np.random.default_rng(77)
        Q = rng.integers(0, CARD, (2000, NG))
        probe = (Q, bench.score(Q, tabular=False))
        e = {'prior_rho': float(spearmanr(pri.predict(Q), probe[1]).statistic)}
        for b in BUDGETS:
            gens = b // 20
            plain = policies.run(bench, R=400, pop=20, gens=gens, k=K, pm=PM, seed=1)
            rec = {'plain': float(plain['best'].mean()),
                   'plain_se': float(plain['best'].std() / np.sqrt(400))}
            for mode, pr in (('scratch', None), ('explore', None), ('transfer', pri)):
                v = loop(bench, b, mode, prior=pr, seed=3, probe=probe)
                rec[mode] = v['best']
                rec[mode + '_se'] = v['se']
                rec[mode + '_fid'] = v['fidelity']
                rec[mode + '_delta'] = v['best'] - rec['plain']
            e[b] = rec
        out[ds] = e
    json.dump(out, open(f'{common.RES}/sitl.json', 'w'), indent=1)

    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Model-in-the-loop search, at an identical number of \emph{trained}',
         r'architectures. The model is a ridge genuinely refitted on what the loop has',
         r'measured. Started from scratch it is worse than training every individual: the',
         r'measured set is whatever the loop chose, so the model never becomes good enough to',
         r'choose with. Started from the surrogate fitted on the \emph{other dataset}, which',
         r'costs no label here, it is better at every budget and most at the smallest.}',
         r'\label{tab:sitl}', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lcccccc}', r'\toprule',
         r'& \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{CIFAR-100} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         r'Trained architectures & 120 & 260 & 520 & 120 & 260 & 520 \\', r'\midrule']
    NAMES = [('plain', 'GA, every individual trained'),
             ('scratch', 'Model in the loop, from scratch'),
             ('explore', r'\quad $+$ random exploration'),
             ('transfer', r'\quad $+$ transferred prior')]
    for key, name in NAMES:
        vals = [f'{out[ds][b][key]:.2f}' for ds in common.DATASETS for b in BUDGETS]
        L.append(rf'{name} & ' + ' & '.join(vals) + r' \\')
    L.append(r'\midrule')
    vals = [rf'\textbf{{{out[ds][b]["transfer_delta"]:+.2f}}}'
            for ds in common.DATASETS for b in BUDGETS]
    L.append(r'\textbf{Transferred prior $-$ GA} & ' + ' & '.join(vals) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_sitl.tex', 'w').write('\n'.join(L) + '\n')

    for ds, e in out.items():
        print(f'== {ds} (transferred prior ranks the space at rho={e["prior_rho"]:.3f}) ==')
        for b in BUDGETS:
            r_ = e[b]
            print(f'  budget {b:3d}: plain={r_["plain"]:6.2f} | scratch={r_["scratch"]:6.2f} '
                  f'({r_["scratch_delta"]:+.2f}) | explore={r_["explore"]:6.2f} '
                  f'({r_["explore_delta"]:+.2f}) | transfer={r_["transfer"]:6.2f} '
                  f'({r_["transfer_delta"]:+.2f})')
            print(f'            end-of-run model fidelity: scratch={r_["scratch_fid"]:.3f} '
                  f'explore={r_["explore_fid"]:.3f} transfer={r_["transfer_fid"]:.3f}')


if __name__ == '__main__':
    main()
