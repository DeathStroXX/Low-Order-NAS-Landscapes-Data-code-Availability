"""Step 9 -- can a free warm start out-run a weak selection signal?

The pipeline selected on a LoRA-tuned 7B predictor whose rank fidelity against the 3-epoch
reference was rho = 0.43 on the earlier run set and is 0.58 (CIFAR-10) and 0.33 (CIFAR-100)
on the October 2026 record (Section IV, results/regret.json). The obvious remedy is a better predictor. This asks
whether a cheaper one works: leave the signal as bad as it is, and seed the initial population
from free surrogate screening instead.

The weak signal is emulated by corrupting the benchmark's value to an achieved rank fidelity
on a grid that contains both measured predictor levels; `predictor_level` in the output names
the level each dataset's predictor actually has. That emulation is legitimate here, unlike inside a model-in-the-loop search, because the
predictor's fidelity was measured externally and does not depend on which architectures the
search chooses to visit.

Writes results/warmstart.json, tables/tab_warmstart.tex.
"""
import json
import numpy as np
import common, fractalbench as fb, policies
from surrogate import CARD, NG

R = 600
SCREEN = 50000
RHOS = [0.33, 0.43, 0.58, 0.70, 0.93]


def main():
    out = {}
    for ds in common.DATASETS:
        bench = fb.Bench(ds)
        rng = np.random.default_rng(5)
        cand = rng.integers(0, CARD, (SCREEN, NG))
        sc = bench.score(cand, tabular=False)
        seed_pop = cand[np.argsort(-sc)[:20]]
        e = {}
        for tag, kw in (('asrun', dict(k=3, pm=0.2)), ('tuned', dict(k=10, pm=0.08))):
            clean = float(policies.run(bench, R=R, seed=1, **kw)['best'].mean())
            clean_warm = float(policies.run(bench, R=R, seed=1, init=seed_pop,
                                            **kw)['best'].mean())
            cell = {'clean': clean, 'clean_warm': clean_warm}
            for rh in RHOS:
                nz = bench.noise_for_rho(rh)
                noisy = float(policies.run(bench, R=R, seed=1, noise=nz, **kw)['best'].mean())
                warm = float(policies.run(bench, R=R, seed=1, noise=nz, init=seed_pop,
                                          **kw)['best'].mean())
                cell[str(rh)] = {
                    'noisy': noisy, 'warm': warm,
                    'loss': noisy - clean, 'loss_warm': warm - clean,
                    'recovered': (warm - noisy) / max(1e-9, clean - noisy)}
            e[tag] = cell
        e['predictor_level'] = '%.2f' % round(json.load(open(f'{common.RES}/regret.json'))[ds]['rho']['predictor'], 2)
        assert e['predictor_level'] in [str(r) for r in RHOS], e['predictor_level']
        out[ds] = e
    json.dump(out, open(f'{common.RES}/warmstart.json', 'w'), indent=1)

    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{A free warm start against a weak selection signal. The GA is run with a',
         r'selection signal corrupted to the rank fidelity shown; the warm-started arm differs',
         r'only in that its initial population is the best 20 of 50{,}000 candidates screened',
         r"for free by the genotype surrogate. Bold: the fidelity of the pipeline's own LLM",
         r'predictor on each dataset ($\rho=0.58$ and $0.33$).}',
         r'\label{tab:warmstart}', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lcccc}', r'\toprule',
         r'& \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{CIFAR-100} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Selection signal & Cold & Warm & Cold & Warm \\', r'\midrule']
    a, b = out['cifar10']['asrun'], out['cifar100']['asrun']
    L.append(rf'Exact ($\rho=1$) & {a["clean"]:.2f} & {a["clean_warm"]:.2f} & '
             rf'{b["clean"]:.2f} & {b["clean_warm"]:.2f} \\')
    for rh in RHOS:
        ra, rb = a[str(rh)], b[str(rh)]
        bf = lambda on: (lambda v: rf'\textbf{{{v:.2f}}}') if on else (lambda v: f'{v:.2f}')
        ba = bf(str(rh) == out['cifar10']['predictor_level'])
        bb = bf(str(rh) == out['cifar100']['predictor_level'])
        L.append(rf'$\rho={rh}$ & {ra["noisy"]:.2f} & {ba(ra["warm"])} & '
                 rf'{rb["noisy"]:.2f} & {bb(rb["warm"])} \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_warmstart.tex', 'w').write('\n'.join(L) + '\n')

    for ds, e in out.items():
        print(f'== {ds} ==')
        for tag, c in e.items():
            if tag == 'predictor_level':
                continue
            print(f'  [{tag}] clean={c["clean"]:.2f} clean+warm={c["clean_warm"]:.2f}')
            for rh in RHOS:
                v = c[str(rh)]
                print(f'     rho={rh}: cold={v["noisy"]:.2f} warm={v["warm"]:.2f} '
                      f'loss {v["loss"]:+.2f} -> {v["loss_warm"]:+.2f} '
                      f'(recovers {100*v["recovered"]:.0f}%)')


if __name__ == '__main__':
    main()
