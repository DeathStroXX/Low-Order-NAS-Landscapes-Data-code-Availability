"""The free genotype surrogate against the zero-cost proxy literature, on NAS-Bench-201.

Section~V of the paper claims that the cheapest possible ranker -- a ridge over gene
indicators and their pairwise interactions -- is worth more than the trained proxies our own
pipeline could afford. The obvious objection is that the field already has a family of
rankers that cost a fraction of an epoch: the zero-cost (ZC) proxies. This script answers
that objection with the reference numbers rather than with an argument.

Data
----
`data/zc_nasbench201.json.xz`: NAS-Bench-Suite-Zero~\\cite{krishnakumar2022nasbenchsuitezero}
as released by its authors -- 13 proxies for all 15,625 NAS-Bench-201 cells on three
datasets, each with the score *and* the wall-clock seconds it took to evaluate. Nothing in it
is recomputed here; the file is used exactly as published, so the proxies are represented by
their authors' own implementations and their authors' own measurement protocol.

`data/nb201_all.pickle`: the accuracy table, as in `s35_nb201.py`. The target stays what it
is everywhere else in the paper -- final test accuracy -- so the numbers here are comparable
to the NAS-Bench-201 fidelity numbers of `s35_nb201.py` line by line.

The join between the two files is by architecture, and it is checked rather than assumed: the
proxy file keys a cell by six operation indices in NASLib's edge order, which is not the order
of the benchmark's own architecture string. The decoding is verified by requiring that the
accuracy the proxy file carries for every one of the 15,625 cells equals the accuracy the
benchmark table carries for the cell the decoding maps it to, to the last digit. Any mismatch
aborts the script.

The three comparisons
---------------------
1. **Label-free.** A ZC proxy needs no label of any kind. The surrogate's label-free form is
   the transfer of Section~V: fit on another dataset, apply with zero target labels. This is
   the comparison that decides whether the free ranker has any standing at all, and it is run
   on identical held-out pools.
2. **Label-cheap.** The surrogate fitted on N target labels, N from 100 to 1,000, against
   every proxy. Reported in rank fidelity and in the decision metric -- regret at a shortlist
   of k, and k*, the smallest shortlist whose regret stays under a tenth of an accuracy point.
3. **Cost.** What each ranker costs to apply to a pool of candidates, in the proxy file's own
   measured seconds. A proxy is cheap per candidate but not free, and screening a large pool
   with one costs real GPU time; the surrogate's cost per candidate is a dot product.

Fusion and ensembles are included so the comparison is not a straw man: the rank-mean of all
13 proxies, the rank-mean of the best three, and the surrogate fused with the best single
proxy at the paper's fixed 2:1 weight.

4. **Out of vision.** The proxy file also covers three further tasks on 199 of the same
   architectures, one of which -- NinaPro, surface electromyography gesture recognition -- is
   not an image-classification problem at all. A prior fitted on CIFAR-100 image labels and
   applied to it with no label of its own is the hardest test of "a property of the space,
   not of the dataset" available here, and it is run with the proxies alongside.

Outputs
-------
results/zc_nb201.json      every number below
results/zc_environment.json  provenance of the inputs
../tables/tab_zc.tex       Table: the head-to-head, main paper
../tables/tab_zc_sm.tex    Table: all 13 proxies, all three datasets, supplement
../figures/fig_zc.pdf      regret against shortlist size, and cost against fidelity
"""
import hashlib, itertools, json, lzma, os, pickle, re, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau

import common
from s35_nb201 import DS, SHORT, fit, regret, encode, PAIRS, N_CELLS

#: NASLib's operation order for the proxy file's integer codes.
ZC_OPS = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
#: The proxy file lists the six cell edges in NASLib's EDGE_LIST order,
#: ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4)); the benchmark's architecture string lists them in
#: ((1,2),(1,3),(2,3),(1,4),(2,4),(3,4)) order. Verified exactly, see `load_zc`.
EDGE_PERM = (0, 1, 3, 2, 4, 5)

#: the 13 proxies of NAS-Bench-Suite-Zero. `params` and `flops` are the two the benchmark
#: table already supplies; the other eleven need a forward or a
#: forward-backward pass over one minibatch.
PROXIES = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
           'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']
#: proxies whose published convention is that a *smaller* score is better. All 13 are used
#: with the sign that maximises rank correlation on CIFAR-10, which is decided here once and
#: recorded, so no proxy is penalised for a sign convention.
NICE = {'epe_nas': 'EPE-NAS', 'fisher': 'Fisher', 'flops': 'FLOPs', 'grad_norm': 'GradNorm',
        'grasp': 'GraSP', 'jacov': 'JacobCov', 'l2_norm': r'$\ell_2$ norm', 'nwot': 'NWOT',
        'params': 'Params', 'plain': 'Plain', 'snip': 'SNIP', 'synflow': 'SynFlow',
        'zen': 'Zen'}

KS = [1, 5, 10, 20, 50, 100, 200, 500]
NFIT = 1000                     #: labels for the surrogate, as in `s35_nb201.py`
NSEED = 20                      #: the same 20 splits as s35_nb201.py
NS = [100, 250, 500, 1000]
TOL = 0.1                       #: the tenth of a point that defines k*


# --------------------------------------------------------------------------- loading
def _open_maybe_xz(path):
    return lzma.open(path, 'rt') if path.endswith('.xz') else open(path)


#: the one instruction a reader needs when a proxy file is absent. The archive does not
#: redistribute these three files: they are NAS-Bench-Suite-Zero exactly as its authors
#: released it, and every number this script derives from them is already in
#: analysis/results/zc_nb201.json, so the download is needed only to recompute them.
HOWTO = """
data/%s(.xz) is missing.

It is NAS-Bench-Suite-Zero as released by its authors (Krishnakumar et al., NAS-Bench-Suite-
Zero: Accelerating Research on Zero Cost Proxies, NeurIPS 2022 Datasets and Benchmarks):
13 zero-cost proxies for every architecture of NAS-Bench-101, -201 and -301, each with the
score and the seconds the measurement took. This archive does not redistribute it.

To fetch it, from a NASLib checkout (https://github.com/automl/NASLib):

    bash scripts/zc/bash_scripts/download_nbs_zero.sh %s

then move naslib/data/%s into this project's data/ directory. The file may be left
xz-compressed (data/%s.xz); every script here reads either form.

You do not need it to check the paper: every number the manuscript quotes from these files is
in analysis/results/zc_nb201.json, zc_spaces.json and encoding.json, the tables and figures are
regenerated from those by s46_tables.py, and s51_verify_claims.py checks the manuscript against
them. The download is needed only to recompute the proxy comparison from the proxy scores.
"""


def zc_missing(fname, space):
    """Exit with the one instruction that resolves a missing proxy file."""
    sys.exit(HOWTO % (fname, space, fname, fname))


def zc_path():
    for name in ['zc_nasbench201.json.xz', 'zc_nasbench201.json']:
        p = os.path.join(common.DATA, name)
        if os.path.exists(p):
            return p
    zc_missing('zc_nasbench201.json', 'nb201')


def load_zc():
    """Read the proxy file and align it to the accuracy table, verifying the alignment.

    The row order is `s35_nb201.load()`'s row order -- the key order of the accuracy table --
    and not the proxy file's. That matters: every split in this file is drawn with the same
    seeds as in `s35_nb201.py`, so sharing the row order is what makes the surrogate's
    fidelity and regret here the *same numbers* as `s35_nb201.py` reports rather than a second
    of them. `s51_verify_claims.py` checks that they agree exactly.

    Returns (P, Y, S, T, n, nonfinite, path) where P is the n x 6 gene matrix in the paper's
    own operation coding, Y[d] the final test accuracy, S[d][proxy] the proxy score and
    T[d][proxy] the measured seconds, all on that one row order.
    """
    from s35_nb201 import load as load35, OI
    keys, P, Y, _cost = load35()
    n = len(keys)

    path = zc_path()
    with _open_maybe_xz(path) as fh:
        raw = json.load(fh)
    for d in DS:
        if d not in raw:
            sys.exit('the proxy file does not contain %s' % d)
    if len(raw[DS[0]]) != n:
        sys.exit('proxy file has %d cells, accuracy table has %d'
                 % (len(raw[DS[0]]), n))

    # proxy-file key (six operation codes, NASLib edge order) -> the paper's gene vector
    zc_by_genes = {}
    for k in raw[DS[0]]:
        code = tuple(int(x) for x in re.findall(r'-?\d+', k))
        if len(code) != 6:
            sys.exit('unexpected architecture key in the proxy file: %r' % k)
        names = [ZC_OPS[c] for c in code]
        toks = tuple(names[EDGE_PERM[j]] for j in range(6))
        zc_by_genes[tuple(OI[t] for t in toks)] = k
    if len(zc_by_genes) != n:
        sys.exit('the decoding is not a bijection: %d distinct cells from %d keys'
                 % (len(zc_by_genes), n))

    # the alignment check, on every cell: the validation accuracy the proxy file carries must
    # equal the validation accuracy the benchmark table carries for the cell we decoded it to.
    with open(os.path.join(common.DATA, 'nb201_all.pickle'), 'rb') as fh:
        tab = pickle.load(fh)
    S = {d: {m: np.zeros(n) for m in PROXIES} for d in DS}
    T = {d: {m: np.zeros(n) for m in PROXIES} for d in DS}
    worst = 0.0
    for i, arch in enumerate(keys):
        k = zc_by_genes.get(tuple(int(x) for x in P[i]))
        if k is None:
            sys.exit('architecture %r has no row in the proxy file' % arch)
        worst = max(worst, abs(tab[arch]['cifar10-valid']['eval_acc1es']
                               - raw['cifar10'][k]['val_accuracy']))
        for d in DS:
            e = raw[d][k]
            for m in PROXIES:
                v = e[m]
                S[d][m][i] = v['score'] if isinstance(v, dict) else v
                T[d][m][i] = v.get('time', np.nan) if isinstance(v, dict) else np.nan
    if worst > 1e-6:
        sys.exit('alignment check failed: worst accuracy mismatch %.8f' % worst)
    print('alignment verified on all %d cells (worst mismatch %.1e)' % (n, worst))
    # a few proxies return non-finite scores on degenerate cells (an all-`none` cell has no
    # gradient); the published convention is to rank those last, which is what a large
    # negative sentinel does after the sign is fixed.
    nonfinite = {}
    for d in DS:
        for m in PROXIES:
            bad = ~np.isfinite(S[d][m])
            if bad.any():
                nonfinite.setdefault(d, {})[m] = int(bad.sum())
    return P, Y, S, T, n, nonfinite, path


def fix_signs(S, Y):
    """Give every proxy the sign under which larger is better, decided on CIFAR-10 once.

    Published conventions differ (JacobCov and GraSP are reported negated in some papers and
    not in others), so rather than trusting a convention we measure it, on one dataset, and
    apply the same sign to all three. Recorded in the output.
    """
    sign = {}
    for m in PROXIES:
        v = S['cifar10'][m]
        ok = np.isfinite(v)
        r = spearmanr(v[ok], Y['cifar10'][ok]).correlation
        sign[m] = -1.0 if r < 0 else 1.0
    out = {}
    for d in S:
        out[d] = {}
        for m in PROXIES:
            v = sign[m] * S[d][m].astype(float)
            bad = ~np.isfinite(v)
            if bad.any():
                v = v.copy()
                v[bad] = np.nanmin(v[~bad]) - 1e6
            out[d][m] = v
    return out, sign


# --------------------------------------------------------------------------- helpers
def ranks(x):
    """Average ranks, 0 for the worst. Used for every fusion, as in the paper's 2:1 rule."""
    from scipy.stats import rankdata
    return rankdata(x)


def kstar(reg_by_k, tol=TOL):
    ks = [k for k in KS if reg_by_k[k] <= tol]
    return min(ks) if ks else None


def summarise(pred, y, out):
    """rho, tau and regret at every k, for one ranker on one held-out pool."""
    out['rho'].append(spearmanr(pred, y).correlation)
    out['tau'].append(kendalltau(pred, y).correlation)
    for k in KS:
        out['rg'][k].append(regret(pred, y, k))


def blank():
    return {'rho': [], 'tau': [], 'rg': {k: [] for k in KS}}


def collapse(acc):
    o = {'rho': float(np.mean(acc['rho'])), 'rho_sd': float(np.std(acc['rho'])),
         'tau': float(np.mean(acc['tau'])),
         'regret': {str(k): float(np.mean(v)) for k, v in acc['rg'].items()},
         'regret_sd': {str(k): float(np.std(v)) for k, v in acc['rg'].items()}}
    o['k_star'] = kstar({k: float(np.mean(v)) for k, v in acc['rg'].items()})
    return o


def main():
    P, Y, S_raw, T, n, nonfinite, path = load_zc()
    S, sign = fix_signs(S_raw, Y)

    out = {'n_arch': n, 'datasets': DS, 'proxies': PROXIES, 'n_seeds': NSEED,
           'n_fit': NFIT, 'ks': KS, 'tol': TOL, 'signs': sign,
           'nonfinite': nonfinite, 'source': os.path.basename(path),
           'edge_perm': list(EDGE_PERM), 'zc_ops': ZC_OPS}

    # ------------------------------------------------------------------ one pass, shared splits
    # Every ranker below is scored on the same 20 held-out pools of 14,625 architectures, and
    # the surrogate is fitted on the 1,000 complementary labels. A proxy needs no fit, so for
    # a proxy the split only selects the pool -- which is the point: the pool must be identical
    # or the regrets are not comparable.
    acc = {('surrogate', str(N)): blank() for N in NS}
    for m in PROXIES:
        acc[('proxy', m)] = blank()
    acc[('fused', 'all13')] = blank()
    acc[('fused', 'top3')] = blank()
    acc[('fused', 'surrogate+best')] = blank()
    acc[('fused', 'transfer+best')] = blank()
    acc[('proxy_picked', 'source')] = blank()
    acc[('proxy_picked', 'fit')] = blank()
    acc[('fused', 'transfer+picked')] = blank()
    acc[('fused', 'surrogate+picked')] = blank()
    acc[('random', '')] = blank()
    for src in DS:
        acc[('transfer', src)] = blank()

    # Which proxy to use has to be decided with labels somebody actually has, and the two
    # metrics disagree sharply: on CIFAR-10 NWOT ranks best ($\rho$) while SynFlow leaves the
    # better shortlist ($r_{10}$). Three selection rules are therefore kept apart and all
    # three are reported.
    #
    #   `best_rho`     : highest rank fidelity, measured over the whole table. This is the
    #                    rule the literature's headline tables use.
    #   `pick_source`  : lowest regret at ten on the *source* dataset of the transfer -- the
    #                    only labels a label-free deployment on the target actually has.
    #   `pick_fit`     : lowest regret at ten on the 1,000 labels the surrogate was fitted on.
    #
    # An oracle that picks the proxy by its regret on the target pool is also recorded, as the
    # ceiling the rules are trying to reach, and is never used as a comparison baseline.
    whole = {d: {m: spearmanr(S[d][m], Y[d]).correlation for m in PROXIES} for d in DS}
    best_proxy = {d: max(PROXIES, key=lambda m: whole[d][m]) for d in DS}
    top3 = {d: sorted(PROXIES, key=lambda m: -whole[d][m])[:3] for d in DS}
    out['whole_table_rho'] = {d: {m: float(v) for m, v in whole[d].items()} for d in DS}
    out['best_proxy'] = best_proxy
    out['best_proxy_rule'] = 'highest rho over the whole table'
    out['top3_proxies'] = top3

    # the source a label-free surrogate would be transferred from, fixed before any split:
    # the hardest other dataset, which is the rule Section V states.
    hardest_other = {d: min([s_ for s_ in DS if s_ != d], key=lambda s_: Y[s_].mean())
                     for d in DS}
    out['transfer_source_rule'] = {
        'rule': 'fit on the hardest other dataset (lowest mean accuracy)',
        'source': hardest_other,
        'mean_acc': {d: float(Y[d].mean()) for d in DS}}

    # `pick_source`: the proxy with the lowest regret at ten on the source dataset, over the
    # same 20 splits, using only the source's labels. Decided once, before the target is seen.
    pick_source, pick_fit = {}, {}
    for d in DS:
        src = hardest_other[d]
        rg_src, rg_fit = {m: [] for m in PROXIES}, {m: [] for m in PROXIES}
        for sd in range(NSEED):
            idx = np.random.default_rng(sd).permutation(n)
            tr, te = idx[:NFIT], idx[NFIT:]
            for m in PROXIES:
                rg_src[m].append(regret(S[src][m][te], Y[src][te], 10))
                rg_fit[m].append(regret(S[d][m][tr], Y[d][tr], 10))
        pick_source[d] = min(PROXIES, key=lambda m: float(np.mean(rg_src[m])))
        pick_fit[d] = min(PROXIES, key=lambda m: float(np.mean(rg_fit[m])))
    out['proxy_selection'] = {
        'best_rho': best_proxy,
        'pick_source': pick_source,
        'pick_fit': pick_fit,
        'rules': {'best_rho': 'highest rho over the whole table',
                  'pick_source': 'lowest regret at ten on the transfer source dataset',
                  'pick_fit': 'lowest regret at ten on the 1,000 fitted labels'}}

    per_ds, paired = {}, {}
    for d in DS:
        A = {k: blank() for k in acc}
        y_all = Y[d]
        for sd in range(NSEED):
            g = np.random.default_rng(sd)
            idx = g.permutation(n)
            tr, te = idx[:NFIT], idx[NFIT:]
            yt = y_all[te]
            # the surrogate at each label budget, nested so the budgets are comparable
            preds_sur = {}
            for N in NS:
                p = fit(P[tr[:N]], y_all[tr[:N]])(P[te])
                preds_sur[N] = p
                summarise(p, yt, A[('surrogate', str(N))])
            # the proxies, on the same pool
            rk = {}
            for m in PROXIES:
                v = S[d][m][te]
                summarise(v, yt, A[('proxy', m)])
                rk[m] = ranks(v)
            summarise(np.mean([rk[m] for m in PROXIES], axis=0), yt, A[('fused', 'all13')])
            summarise(np.mean([rk[m] for m in top3[d]], axis=0), yt, A[('fused', 'top3')])
            # the paper's fixed 2:1 rank fusion, surrogate twice the weight of the proxy
            rs = ranks(preds_sur[NFIT])
            summarise((2 * rs + rk[best_proxy[d]]) / 3.0, yt, A[('fused', 'surrogate+best')])
            summarise(S[d][pick_source[d]][te], yt, A[('proxy_picked', 'source')])
            summarise(S[d][pick_fit[d]][te], yt, A[('proxy_picked', 'fit')])
            summarise((2 * rs + rk[pick_fit[d]]) / 3.0, yt,
                      A[('fused', 'surrogate+picked')])
            summarise(g.standard_normal(len(te)), yt, A[('random', '')])
            # label-free: the surrogate fitted on another dataset's 1,000 labels
            for src in DS:
                if src == d:
                    continue
                p = fit(P[tr], Y[src][tr])(P[te])
                summarise(p, yt, A[('transfer', src)])
                if src == hardest_other[d]:
                    # the two free rankers put together, at the paper's fixed 2:1 weight,
                    # with the proxy chosen by each of the two label-honest rules
                    summarise((2 * ranks(p) + rk[best_proxy[d]]) / 3.0, yt,
                              A[('fused', 'transfer+best')])
                    summarise((2 * ranks(p) + rk[pick_source[d]]) / 3.0, yt,
                              A[('fused', 'transfer+picked')])
        per_ds[d] = {'|'.join(k): collapse(v) for k, v in A.items() if v['rho']}

        # paired differences over the 20 shared splits, which is the only way these rankers
        # can be compared: the same pool, the same split, the two regrets subtracted.
        def pair(a, b, k=10):
            x = np.array(A[a]['rg'][k]) - np.array(A[b]['rg'][k])
            lo, hi = np.percentile(
                [np.mean(x[np.random.default_rng(i).integers(0, len(x), len(x))])
                 for i in range(2000)], [2.5, 97.5])
            return {'mean': float(x.mean()), 'ci': [float(lo), float(hi)],
                    'wins': int((x < 0).sum()), 'n': int(len(x))}

        bp = ('proxy', best_proxy[d])
        ps = ('proxy_picked', 'source')
        paired[d] = {
            'surrogate1000_minus_bestproxy': pair(('surrogate', '1000'), bp),
            'transfer_minus_bestproxy': pair(('transfer', hardest_other[d]), bp),
            'fused_transfer_best_minus_bestproxy': pair(('fused', 'transfer+best'), bp),
            'fused_transfer_best_minus_transfer': pair(('fused', 'transfer+best'),
                                                       ('transfer', hardest_other[d])),
            'bestproxy_minus_random': pair(bp, ('random', '')),
            'rankmean13_minus_bestproxy': pair(('fused', 'all13'), bp),
            # the contrasts that matter, against the proxy a practitioner would really pick
            'transfer_minus_pickedproxy': pair(('transfer', hardest_other[d]), ps),
            'fused_picked_minus_pickedproxy': pair(('fused', 'transfer+picked'), ps),
            'fused_picked_minus_transfer': pair(('fused', 'transfer+picked'),
                                                ('transfer', hardest_other[d])),
            'pickedproxy_minus_random': pair(ps, ('random', ''))}

    out['per_dataset'] = per_ds
    out['paired'] = paired

    # ------------------------------------------------------------------ the label-free verdict
    # For each dataset: the best proxy's regret at a shortlist of ten against the best
    # transferred surrogate's, both with zero target labels.
    lf = {}
    for d in DS:
        bt = 'transfer|' + hardest_other[d]
        bp = 'proxy|' + best_proxy[d]
        lf[d] = {
            'best_transfer_source': bt.split('|')[1],
            'transfer_rho': per_ds[d][bt]['rho'],
            'transfer_regret10': per_ds[d][bt]['regret']['10'],
            'transfer_kstar': per_ds[d][bt]['k_star'],
            'best_proxy': best_proxy[d],
            'proxy_rho': per_ds[d][bp]['rho'],
            'proxy_regret10': per_ds[d][bp]['regret']['10'],
            'proxy_kstar': per_ds[d][bp]['k_star'],
            'proxy_wins_rho': per_ds[d][bp]['rho'] > per_ds[d][bt]['rho'],
            'proxy_wins_regret10': (per_ds[d][bp]['regret']['10']
                                    < per_ds[d][bt]['regret']['10']),
            'fused_rho': per_ds[d]['fused|transfer+best']['rho'],
            'fused_regret10': per_ds[d]['fused|transfer+best']['regret']['10'],
            'fused_kstar': per_ds[d]['fused|transfer+best']['k_star'],
            # the proxy a practitioner would pick with the labels they have, and the fusion
            'picked_proxy': pick_source[d],
            'picked_rho': per_ds[d]['proxy_picked|source']['rho'],
            'picked_regret10': per_ds[d]['proxy_picked|source']['regret']['10'],
            'picked_kstar': per_ds[d]['proxy_picked|source']['k_star'],
            'picked_wins_regret10': (per_ds[d]['proxy_picked|source']['regret']['10']
                                     < per_ds[d][bt]['regret']['10']),
            'fused_picked_rho': per_ds[d]['fused|transfer+picked']['rho'],
            'fused_picked_regret10': per_ds[d]['fused|transfer+picked']['regret']['10'],
            'fused_picked_kstar': per_ds[d]['fused|transfer+picked']['k_star'],
            # the ceiling, selected on the target pool; never used as a baseline
            'oracle_proxy_regret10': min(PROXIES,
                                         key=lambda m: per_ds[d]['proxy|' + m]['regret']['10']),
            'oracle_regret10': min(per_ds[d]['proxy|' + m]['regret']['10']
                                   for m in PROXIES)}
    out['label_free'] = lf

    # ------------------------------------------------------------------ out of vision
    # The proxy file carries three further tasks on a 199-architecture subset, including one
    # that is not image classification (NinaPro, surface electromyography gesture
    # recognition). The surrogate is fitted on 1,000 labels of a *vision* dataset and applied
    # with no label of the target task; the proxies are the values the file records for that
    # task. The pool is 199 architectures, so k runs only to 50 and the intervals are wide --
    # this is a direction check, not a headline.
    EXTRA = ['svhn', 'scifar100', 'ninapro']
    with _open_maybe_xz(path) as fh:
        raw_extra = json.load(fh)
    extra = {}
    from s35_nb201 import OI
    for task in EXTRA:
        if task not in raw_extra:
            continue
        sub = raw_extra[task]
        rows, yv, sv = [], [], {m: [] for m in PROXIES}
        for k, e in sub.items():
            code = tuple(int(x) for x in re.findall(r'-?\d+', k))
            names = [ZC_OPS[c] for c in code]
            toks = tuple(names[EDGE_PERM[j]] for j in range(6))
            rows.append([OI[t] for t in toks])
            yv.append(e['val_accuracy'])
            for m in PROXIES:
                v = e[m]
                sv[m].append(v['score'] if isinstance(v, dict) else v)
        Pe = np.array(rows, dtype=np.int16)
        ye = np.array(yv, float)
        ks_small = [k for k in KS if k <= 50]
        ent = {'n_arch': len(ye), 'acc_mean': float(ye.mean()),
               'acc_max': float(ye.max()), 'ks': ks_small}
        # the proxies, with the signs already fixed on CIFAR-10
        for m in PROXIES:
            v = sign[m] * np.asarray(sv[m], float)
            bad = ~np.isfinite(v)
            if bad.any():
                v = v.copy()
                v[bad] = np.nanmin(v[~bad]) - 1e6
            ent['proxy|' + m] = {
                'rho': float(spearmanr(v, ye).correlation),
                'regret': {str(k): regret(v, ye, k) for k in ks_small}}
        # the surrogate, fitted on each vision dataset's labels, zero labels of this task
        for src in DS:
            rho, rg = [], {k: [] for k in ks_small}
            for sd in range(NSEED):
                tr = np.random.default_rng(sd).permutation(n)[:NFIT]
                p = fit(P[tr], Y[src][tr])(Pe)
                rho.append(spearmanr(p, ye).correlation)
                for k in ks_small:
                    rg[k].append(regret(p, ye, k))
            ent['transfer|' + src] = {
                'rho': float(np.mean(rho)), 'rho_sd': float(np.std(rho)),
                'regret': {str(k): float(np.mean(v)) for k, v in rg.items()}}
        rnd = {k: [] for k in ks_small}
        for sd in range(NSEED):
            g = np.random.default_rng(1000 + sd)
            for k in ks_small:
                rnd[k].append(regret(g.standard_normal(len(ye)), ye, k))
        ent['random|'] = {'rho': 0.0,
                          'regret': {str(k): float(np.mean(v)) for k, v in rnd.items()}}
        ent['best_proxy'] = max(PROXIES, key=lambda m: ent['proxy|' + m]['rho'])
        ent['best_transfer'] = max(DS, key=lambda d_: ent['transfer|' + d_]['rho'])
        extra[task] = ent
    out['extra_tasks'] = extra
    del raw_extra

    # ------------------------------------------------------------------ cost, in measured seconds
    # The proxy file records the seconds each measurement took. Screening a pool costs that
    # per candidate; the surrogate costs one dot product per candidate and its labels are
    # already paid for by the search that produced them.
    cost = {}
    for d in DS:
        cost[d] = {}
        for m in PROXIES:
            t = T[d][m]
            t = t[np.isfinite(t)]
            cost[d][m] = {'sec_per_arch': float(np.mean(t)) if len(t) else None,
                          'sec_pool_14625': float(np.mean(t) * (n - NFIT)) if len(t) else None,
                          'hours_pool_14625': float(np.mean(t) * (n - NFIT) / 3600)
                          if len(t) else None}
    out['cost'] = cost
    # the surrogate's own cost, measured here: fit once, then one pass over the pool
    import time
    t0 = time.perf_counter()
    f = fit(P[:NFIT], Y['cifar10'][:NFIT])
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    f(P[NFIT:])
    t_apply = time.perf_counter() - t0
    out['surrogate_cost'] = {
        'fit_sec': t_fit, 'apply_pool_sec': t_apply,
        'sec_per_arch': t_apply / (n - NFIT),
        'note': 'single CPU core, no GPU; the labels it is fitted on are the search log'}

    # ------------------------------------------------------------------ what a reviewer will ask
    # Does adding the best proxy to the surrogate help, and does adding the surrogate to the
    # best proxy help? Both directions, in the decision metric.
    out['increment'] = {}
    for d in DS:
        s = per_ds[d]['surrogate|1000']
        p = per_ds[d]['proxy|' + best_proxy[d]]
        fz = per_ds[d]['fused|surrogate+best']
        out['increment'][d] = {
            'surrogate_regret10': s['regret']['10'],
            'proxy_regret10': p['regret']['10'],
            'fused_regret10': fz['regret']['10'],
            'fused_vs_surrogate': fz['regret']['10'] - s['regret']['10'],
            'fused_vs_proxy': fz['regret']['10'] - p['regret']['10'],
            'surrogate_kstar': s['k_star'], 'proxy_kstar': p['k_star'],
            'fused_kstar': fz['k_star']}

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'zc_nb201.json'), 'w'), indent=1)
    env = {'zc_file': os.path.basename(path),
           'zc_sha256': _sha(path),
           'zc_reference': 'Krishnakumar et al., NAS-Bench-Suite-Zero, NeurIPS 2022 D&B',
           'accuracy_table': 'nb201_all.pickle (NASLib redistribution of NAS-Bench-201)',
           'target': 'final test accuracy (eval_acc1es), as everywhere else in the paper',
           'alignment_check': 'validation accuracy identical for all 15,625 cells',
           'numpy': np.__version__}
    json.dump(env, open(os.path.join(common.RES, 'zc_environment.json'), 'w'), indent=1)

    emit_table(out)
    emit_sm_table(out)
    emit_figure(out, per_ds)
    print(json.dumps({
        'best_proxy': best_proxy,
        'surrogate_1000_rho': {d: round(per_ds[d]['surrogate|1000']['rho'], 3) for d in DS},
        'best_proxy_rho': {d: round(per_ds[d]['proxy|' + best_proxy[d]]['rho'], 3) for d in DS},
        'surrogate_kstar': {d: per_ds[d]['surrogate|1000']['k_star'] for d in DS},
        'best_proxy_kstar': {d: per_ds[d]['proxy|' + best_proxy[d]]['k_star'] for d in DS},
        'label_free': {d: {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in lf[d].items()} for d in DS},
        'fused_kstar': {d: per_ds[d]['fused|surrogate+best']['k_star'] for d in DS},
        'picked_proxy': pick_source,
        'paired_regret10_transfer_minus_pickedproxy': {
            d: [round(paired[d]['transfer_minus_pickedproxy']['mean'], 3),
                [round(x, 3) for x in paired[d]['transfer_minus_pickedproxy']['ci']]]
            for d in DS},
        'paired_regret10_fusedpicked_minus_pickedproxy': {
            d: [round(paired[d]['fused_picked_minus_pickedproxy']['mean'], 3),
                [round(x, 3) for x in paired[d]['fused_picked_minus_pickedproxy']['ci']]]
            for d in DS},
        'paired_regret10_transfer_minus_bestproxy': {
            d: [round(paired[d]['transfer_minus_bestproxy']['mean'], 3),
                [round(x, 3) for x in paired[d]['transfer_minus_bestproxy']['ci']]]
            for d in DS},
        'extra_tasks': {t: {'n': e['n_arch'],
                            'best_transfer': e['best_transfer'],
                            'transfer_rho': round(e['transfer|' + e['best_transfer']]['rho'], 3),
                            'best_proxy': e['best_proxy'],
                            'proxy_rho': round(e['proxy|' + e['best_proxy']]['rho'], 3)}
                        for t, e in out['extra_tasks'].items()},
        'paired_regret10_fused_minus_bestproxy': {
            d: [round(paired[d]['fused_transfer_best_minus_bestproxy']['mean'], 3),
                [round(x, 3) for x in paired[d]['fused_transfer_best_minus_bestproxy']['ci']]]
            for d in DS}}, indent=1))


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- tables
def emit_table(o):
    """Main paper: the surrogate against the proxy family, in both units, three datasets."""
    d0 = o['per_dataset']
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering')
    A(r'\caption{The free ranker against the zero-cost proxy family on NAS-Bench-201, on '
      r'identical held-out pools of %d architectures. Scores and measurement times for all '
      r'13 proxies are taken unchanged from NAS-Bench-Suite-Zero~\cite{krishnakumar2022nasbenchsuitezero}; '
      r'$\rho$ is against final test accuracy, $r_{10}$ is the accuracy lost by training the '
      r'ten best-ranked candidates instead of the whole pool, and $k^{\star}$ is the smallest '
      r'shortlist whose regret stays under $0.1$ points ($>$500 means no shortlist in the grid '
      r'does). \emph{Zero target labels} compares the two rankers that need none: the best '
      r'proxy, and the surrogate fitted on another dataset. Full table for all 13 proxies: '
      r'Supplementary Table~S19.}' % (o['n_arch'] - o['n_fit']))
    A(r'\label{tab:zc}')
    A(r'\footnotesize')
    A(r'\setlength{\tabcolsep}{3.2pt}')
    A(r'\resizebox{\columnwidth}{!}{%')
    A(r'\begin{tabular}{l ccc ccc}')
    A(r'\toprule')
    A(r'& \multicolumn{3}{c}{$\rho$} & \multicolumn{3}{c}{$r_{10}$ / $k^{\star}$} \\')
    A(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    A(r'& C10 & C100 & IN16 & C10 & C100 & IN16 \\')
    A(r'\midrule')
    A(r'\multicolumn{7}{l}{\emph{Needs labels of the target dataset}} \\')
    for N in [100, 1000]:
        key = 'surrogate|%d' % N
        A(r'\quad Genotype surrogate, %d labels & ' % N
          + ' & '.join('%.3f' % d0[d][key]['rho'] for d in DS) + ' & '
          + ' & '.join(_rk(d0[d][key]) for d in DS) + r' \\')
    A(r'\addlinespace[1pt]')
    A(r'\multicolumn{7}{l}{\emph{Zero target labels}} \\')
    for m in ['synflow', 'nwot', 'jacov', 'params']:
        key = 'proxy|' + m
        A(r'\quad %s & ' % NICE[m]
          + ' & '.join('%.3f' % d0[d][key]['rho'] for d in DS) + ' & '
          + ' & '.join(_rk(d0[d][key]) for d in DS) + r' \\')
    A(r'\quad Rank-mean of all 13 proxies & '
      + ' & '.join('%.3f' % d0[d]['fused|all13']['rho'] for d in DS) + ' & '
      + ' & '.join(_rk(d0[d]['fused|all13']) for d in DS) + r' \\')
    cells_r, cells_k = [], []
    for d in DS:
        src = o['label_free'][d]['best_transfer_source']
        e = d0[d]['transfer|' + src]
        cells_r.append('%.3f' % e['rho'])
        cells_k.append(_rk(e))
    A(r'\quad Surrogate, transferred in & ' + ' & '.join(cells_r) + ' & '
      + ' & '.join(cells_k) + r' \\')
    A(r'\quad \textbf{Transferred surrogate $+$ NWOT (2:1)} & '
      + ' & '.join('%.3f' % d0[d]['fused|transfer+best']['rho'] for d in DS) + ' & '
      + ' & '.join(_rk(d0[d]['fused|transfer+best']) for d in DS) + r' \\')
    A(r'\addlinespace[1pt]')
    A(r'\multicolumn{7}{l}{\emph{Combined, and the floor}} \\')
    A(r'\quad Surrogate $+$ best proxy (2:1) & '
      + ' & '.join('%.3f' % d0[d]['fused|surrogate+best']['rho'] for d in DS) + ' & '
      + ' & '.join(_rk(d0[d]['fused|surrogate+best']) for d in DS) + r' \\')
    A(r'\quad Random shortlist & '
      + ' & '.join('%.3f' % d0[d]['random|']['rho'] for d in DS) + ' & '
      + ' & '.join(_rk(d0[d]['random|']) for d in DS) + r' \\')
    A(r'\bottomrule')
    A(r'\end{tabular}}')
    A(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_zc.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def _rk(e):
    k = e['k_star']
    return '%.2f / %s' % (e['regret']['10'], ('$>$500' if k is None else str(k)))


def emit_sm_table(o):
    d0 = o['per_dataset']
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering')
    A(r'\caption{All 13 zero-cost proxies of NAS-Bench-Suite-Zero~'
      r'\cite{krishnakumar2022nasbenchsuitezero} against the genotype surrogate, on the same '
      r'20 held-out pools of %d NAS-Bench-201 architectures. $\rho$ and Kendall $\tau$ are '
      r'against final test accuracy; $r_k$ is regret at a shortlist of $k$ in accuracy points; '
      r'$k^{\star}$ is the smallest shortlist in $\{1,5,10,20,50,100,200,500\}$ whose mean '
      r'regret is at most $0.1$ points. \emph{sec/arch} is the measurement time the benchmark '
      r'records, averaged over the table, and \emph{pool} is what screening the %d held-out '
      r'architectures with that proxy would cost. Proxy signs are fixed once on CIFAR-10 and '
      r'listed in \texttt{results/zc\_nb201.json}.}'
      % (o['n_arch'] - o['n_fit'], o['n_arch'] - o['n_fit']))
    A(r'\label{tab:zc_sm}')
    A(r'\footnotesize')
    A(r'\setlength{\tabcolsep}{3.5pt}')
    A(r'\resizebox{\textwidth}{!}{%')
    A(r'\begin{tabular}{l rrrr rrrr rrrr r}')
    A(r'\toprule')
    A(r'& \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{CIFAR-100} '
      r'& \multicolumn{4}{c}{ImageNet16-120} & \\')
    A(r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}')
    A(r'Ranker & $\rho$ & $\tau$ & $r_{10}$ & $k^{\star}$'
      r' & $\rho$ & $\tau$ & $r_{10}$ & $k^{\star}$'
      r' & $\rho$ & $\tau$ & $r_{10}$ & $k^{\star}$ & sec/arch \\')
    A(r'\midrule')

    def row(label, key, sec=None):
        cs = []
        for d in DS:
            e = d0[d][key]
            k = e['k_star']
            cs += ['%.3f' % e['rho'], '%.3f' % e['tau'], '%.2f' % e['regret']['10'],
                   ('$>$500' if k is None else str(k))]
        s = '---' if sec is None else ('%.2f' % sec)
        A(label + ' & ' + ' & '.join(cs) + ' & ' + s + r' \\')

    for m in sorted(PROXIES, key=lambda m: -o['whole_table_rho']['cifar10'][m]):
        sec = o['cost']['cifar10'][m]['sec_per_arch']
        row(r'\quad ' + NICE[m], 'proxy|' + m, sec)
    A(r'\addlinespace[2pt]\midrule')
    row(r'\quad Rank-mean, all 13', 'fused|all13',
        sum(o['cost']['cifar10'][m]['sec_per_arch'] or 0 for m in PROXIES))
    row(r'\quad Rank-mean, best 3', 'fused|top3',
        sum(o['cost']['cifar10'][m]['sec_per_arch'] or 0 for m in o['top3_proxies']['cifar10']))
    A(r'\addlinespace[2pt]\midrule')
    for N in NS:
        row(r'\quad Genotype surrogate, %d labels' % N, 'surrogate|%d' % N,
            o['surrogate_cost']['sec_per_arch'])
    for src in DS:
        keys = [d for d in DS if d != src]
        # one row per source, with the two targets it applies to and a dash for itself
        cs = []
        for d in DS:
            if d == src:
                cs += ['---'] * 4
                continue
            e = d0[d]['transfer|' + src]
            k = e['k_star']
            cs += ['%.3f' % e['rho'], '%.3f' % e['tau'], '%.2f' % e['regret']['10'],
                   ('$>$500' if k is None else str(k))]
        A(r'\quad Surrogate fitted on %s, no target label & ' % SHORT[src]
          + ' & '.join(cs) + ' & %.2g' % o['surrogate_cost']['sec_per_arch'] + r' \\')
    row(r'\quad Surrogate $+$ best proxy (2:1)', 'fused|surrogate+best', None)
    row(r'\quad Transferred surrogate $+$ best proxy (2:1)', 'fused|transfer+best', None)
    row(r'\quad Random shortlist', 'random|', 0.0)
    A(r'\bottomrule')
    A(r'\end{tabular}}')
    A(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_zc_sm.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def emit_figure(o, per_ds):
    """One panel: what a shortlist costs in accuracy on CIFAR-100.

    The upper panel shows the rankers the text actually compares -- the two free ones, the
    proxy a practitioner would really pick, their fusion, and the floor -- rather than the
    proxy that happens to win the rank correlation, which is not the one that wins the
    shortlist.

    The companion panel -- mean fidelity against the seconds a ranking costs -- is in the body
    of the paper (s69_body_panels.py). This panel is one column wide.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #: the cost-against-fidelity panel is in the body (Fig. 3b); this is the shortlist panel
    fig, axes = [None, None]
    fig, ax0 = plt.subplots(1, 1, figsize=(3.4, 2.45), constrained_layout=True)
    axes = [ax0]

    d = 'cifar100'
    src = o['transfer_source_rule']['source'][d]
    picked = o['proxy_selection']['pick_source'][d]
    ax = axes[0]
    show = [('transfer|' + src, 'Surrogate transferred in', '#1b6ca8', '-', 'o'),
            ('proxy|synflow', 'SynFlow', '#c7522a', '-', 's'),
            ('proxy|' + picked, 'Proxy picked on source labels', '#d9a23b', '--', 'v'),
            ('fused|transfer+picked', 'Transferred surrogate $+$ proxy', '#2e7d4f', '-', 'D'),
            ('surrogate|1000', 'Surrogate, own 1,000 labels', '#7a5195', '--', '^'),
            ('random|', 'Random shortlist', '#808080', ':', 'x')]
    for key, lab, c, ls, mk in show:
        if key not in per_ds[d]:
            continue
        e = per_ds[d][key]
        y = [max(e['regret'][str(k)], 1e-3) for k in KS]
        ax.plot(KS, y, ls, color=c, marker=mk, ms=3.2, lw=1.4, label=lab)
    ax.axhline(TOL, color='k', lw=0.7, ls='-.')
    ax.text(1.1, TOL * 1.15, r'$0.1$ points', fontsize=5.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('shortlist size $k$', fontsize=7)
    ax.set_ylabel('regret (accuracy points)', fontsize=7)
    ax.set_title('CIFAR-100: what a shortlist gives up', fontsize=7.4, loc='left')
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=.25, lw=.5)
    ax.legend(fontsize=5.0, frameon=False, loc='lower left', handlelength=1.4,
              labelspacing=0.25, borderpad=0.1, handletextpad=0.4)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_zc.' + ext),
                    dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/fig_zc.pdf')


def figure_only():
    """Redraw figures/fig_zc.pdf from results/zc_nb201.json, without the proxy download.

    The proxy scores are NAS-Bench-Suite-Zero's and are not redistributed with this archive,
    so `main()` stops without them. Every quantity the figure draws is already in the result
    file, though, so the figure should not be the one object in the archive that cannot be
    rebuilt from it -- otherwise a change to the drawing code can only be checked by whoever
    has the download. This path reads the result file and redraws.
    """
    with open(os.path.join(common.RES, 'zc_nb201.json')) as fh:
        o = json.load(fh)
    emit_figure(o, o['per_dataset'])


if __name__ == '__main__':
    if '--figure-only' in sys.argv:
        figure_only()
    else:
        main()
