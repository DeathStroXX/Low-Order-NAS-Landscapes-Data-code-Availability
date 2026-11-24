r"""The body-length form of each ledger row, for Table I.

The ledger exists twice. The supplement has room for the full confrontation -- what the
closest prior art establishes, at the length it takes to state fairly -- and that version is
`tables/tab_novelty.tex`. The body has room for about a fifth of that, and the body is where
a reader decides whether the paper is incremental, so the short form cannot simply be the
long one truncated: it has to carry the same comparison in a quarter of the words.

This file holds only that prose. Every *number* in either table still comes from
`analysis/results/` by way of `s49_novelty.py`, which pairs each row here with the evidence
string it computes; nothing numeric is typed here, and the two tables cannot disagree about a
value because only one of them owns any.

The order is the order the claims appear in the manuscript, and it must match `rows()` in
`s49_novelty.py` one for one -- `s49` asserts that it does.
"""

#: (prior art, in one or two sentences; what this paper adds, in two or three)
SHORT = [
    # ---- 1. the exact spectrum
    (r"Exact fANOVA on complete grids: to order eleven on AutoML loss "
     r"landscapes~\cite{pushak2022automl}, to order two on one architecture "
     r"grid~\cite{klein2019tabular}; the categorical closed form in full "
     r"generality~\cite{ferrere2026exact}.",
     r"The same decomposition turned on NAS landscapes, where the answer was not foregone: it "
     r"makes AutoML landscapes \emph{high}-order and both NAS spaces low-order. The apparatus "
     r"is borrowed; the measurement and its consequences are not."),
    # ---- 2. the ceiling
    (r"Cheap predictors are rarely beaten~\cite{white2021powerful,ning2021evaluating} and some "
     r"ruggedness is seed noise~\cite{white2021exploring}; neither is a limit.",
     r"A limit rather than a comparison: because the components are orthogonal, the cumulative "
     r"spectrum caps the $R^{2}$ of \emph{any} degree-$d$ function, however much data it is "
     r"given."),
    # ---- 3. the scale illusion
    (r"On this space correlation length grows with task difficulty and search is more reliable "
     r"on harder tasks~\cite{tao2024landscape,lopes2024operation}; no mechanism is offered.",
     r"The mechanism, which dissolves the effect: easy tasks are compressed against an "
     r"accuracy ceiling, a monotone recalibration collapses the three spectra onto one, and "
     r"where there is no compression it buys almost nothing."),
    # ---- 4. the correction to the recommendation
    (r"Fit the prior on the hardest dataset available and transfer it; it outranks the same "
     r"model fitted on the target's own "
     r"labels~\cite{tao2024landscape,lopes2024operation}.",
     r"True as measured, but mostly a matter of scale: recalibrating the target's own labels "
     r"beats raw-scale transfer at every budget on both metrics, and recalibrating the source "
     r"closes most of that gap."),
    # ---- 5. label complexity
    (r"Protein and RNA fitness functions are Fourier-sparse, so label complexity follows that "
     r"sparsity rather than the size of the space~\cite{brookes2022sparsity}.",
     r"The same bound in NAS, with a consequence that setting does not have: a study that has "
     r"already run owns the labels a degree-2 model of a $10^{18}$-point space needs."),
    # ---- 6. the encoding boundary
    (r"Path encodings beat adjacency encodings, justified by relabelling invariance and "
     r"graceful truncation~\cite{white2020encodings}.",
     r"The spectral reading: a re-encoding adds no information, it lowers the order, and a "
     r"held-out $R^{2}$ measures whether it did---so the scope boundary is testable before a "
     r"budget is spent, and repairable for free."),
    # ---- 7. the decision metric
    (r"Cheap rankers are compared by rank correlation, one space at a "
     r"time~\cite{white2021powerful,abdelfattah2021zerocost,shu2022unifying}.",
     r"The two orderings disagree in both directions, so correlation does not answer what a "
     r"budget asks; we report $k^{\star}$, the smallest shortlist whose regret falls below a "
     r"tenth of a point."),
    # ---- 8. non-identifiability of proxy choice
    (r"A per-space table of which zero-cost proxy wins, computed with the target labels in "
     r"hand~\cite{krishnakumar2022nasbenchsuitezero,shu2022unifying}.",
     r"That table is an oracle quantity: choosing on the only labels a label-free deployment "
     r"has costs most of the advantage, so the deployable object is an untuned fusion rather "
     r"than a choice."),
    # ---- 9. the emulation pitfall
    (r"Price a policy without GPUs by degrading a measured reference to the cheap signal's "
     r"fidelity~\cite{zela2022surrogate,mehta2022nasbenchsuite}.",
     r"The precondition fails where the device is most tempting: inside a model-in-the-loop "
     r"search the fidelity depends on what the policy measures, so the loop must be refitted "
     r"and the shortcut inverts its own sign."),
]
