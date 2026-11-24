# The one confirmation left, pre-registered

Section VIII of the paper measures two policy changes on the released per-architecture
record and on FractalNAS-13 rather than inside a new search. That is the paper's main
remaining weakness and it is named as such in the Limitations. This document fixes, in
advance, exactly what the confirming run will be, what counts as confirmation and what counts
as refutation, so that the result cannot be chosen after it is seen.

It is written to be executed as it stands. `run_confirm.py` in this directory implements the
protocol and `score_confirm.py` scores it against the predictions below. Nothing in the
manuscript depends on the outcome: the paper's claims are stated as measured on the record and
on the benchmark, and this is the test that would overturn them.

*Pointer note.* The claims, arms, seeds, predictions and decision rule below are as
registered and are not edited. Only the section numbers pointing into the manuscript were
updated when the manuscript was restructured; the registered content they point at is
unchanged.

Registered on 2026-10-09, before any run. One GPU; the two arms together are
**24 GPU-hours** at the 100-epoch budget of Section VIII, and **about 90 GPU-hours** if the
520-architecture point of Section VIII is included.

---

## 1. What is being tested

Two claims, each with a number the paper already commits to.

| # | Claim | Where | Predicted |
|---|---|---|---|
| **C1** | The composed budget policy — screen free, train a shortlist for one epoch, allocate by successive halving — beats the flat one-epoch policy at an identical 100-epoch budget. | Sec. VIII, Supplementary Table S4 | **+4.28** (C10), **+7.95** (C100) |
| **C2** | A model-in-the-loop GA seeded with the surrogate transferred from the *other* dataset beats training every individual, at a 120-architecture budget. | Sec. VIII, Supplementary Table S5 | **+4.18** (C10), **+4.76** (C100) |

Both predictions are point estimates from the record. The registered test is on the
**direction and the order of magnitude**, not on the second decimal, because a single run of a
stochastic search cannot resolve a tenth of a point (Sec. IX, *Benchmark versus measurement*).

## 2. Confirmation and refutation, decided now

For each claim and each dataset, with three seeds per arm:

* **Confirmed** if the mean difference is positive and its 95% bootstrap interval excludes
  zero, and the mean is at least **half** the predicted value (≥ +2.14 and +3.98 for C1;
  ≥ +2.09 and +2.38 for C2).
* **Directionally confirmed, magnitude not** if the interval excludes zero but the mean is
  below half the prediction. The paper's claim would then need restating with the measured
  number, which is what the revision would do.
* **Refuted** if the mean difference is negative with an interval excluding zero. The
  corresponding section would be withdrawn and replaced by this result.
* **Unresolved** if the interval contains zero. Three seeds is a weak test and this is the
  likeliest outcome for a difference under one point; it is not evidence for the claim.

A claim is **not** confirmed by one dataset alone. Both are reported whatever happens, and the
raw logs of every arm go into the archive next to the existing 34 runs.

## 3. The protocol

Fixed before the run; `run_confirm.py` takes no argument that could change it.

**Space, data, training.** The 13-gene FractalNet space, data pipeline, optimiser and accuracy
floor of Supplementary Section S3, unchanged, so the result is comparable to the 34 released runs. CIFAR-10
and CIFAR-100, the standard splits. Seeds 4001, 4002, 4003 for every arm; the seed controls
initialisation, data order and every draw the search makes.

**C1 — four arms, 100 training epochs each, per dataset per seed.**

| Arm | Policy |
|---|---|
| `flat` | draw 100 candidates uniformly, train each one epoch, return the best |
| `screen` | screen 800 free candidates with the surrogate, train the best 100 for one epoch |
| `halve` | 73 candidates for one epoch, top 25% for a second, half of those for a third |
| `recipe` | screen 800 free, then allocate the 100 epochs by halving as in `halve` |

The surrogate is fitted on the released record of the *other* dataset only — no label of the
run's own dataset enters the screen — so the screen is free in the sense Sec. VIII means.
Reported outcome: the final accuracy of the returned architecture, trained to the pipeline's
3-epoch reference, the same quantity Supplementary Table S4 reports.

**C2 — three arms, 120 trained architectures each, per dataset per seed.**

| Arm | Policy |
|---|---|
| `train-all` | the GA of the released pipeline, every individual trained, 6 generations of 20 |
| `sitl-seeded` | population 200, the free model ranks all of them, the best 4 per generation are trained, 30 generations; the model is seeded with the transferred surrogate and refitted on the residual of what has been measured |
| `sitl-cold` | the same loop with no prior, fitted only on what it measures |

`sitl-cold` is included because Sec. VIII's mechanism claim — that the prior is the whole of
it — predicts it will *lose* by 7 to 9 points. A result in which both loops lose equally would
refute the mechanism while leaving C2 intact, and that distinction is only visible if the cold
arm is run.

## 4. What is already decided, so that it cannot be decided later

* The surrogate's hyper-parameter is the $\alpha = 10$ of `surrogate.py`, untouched.
* The shortlist sizes, halving fractions and population sizes are the ones above, taken from
  the sections being tested.
* Three seeds, not "three and more if it looks close".
* The 3-epoch reference is the outcome, not the 1-epoch accuracy the policies select on.
* Every arm's log is released, including arms that lose, and the accuracy floor's discard rate
  is reported per arm, because the released distributions are left-truncated by it (Sec. IX).
* If a run crashes, it is rerun with the same seed and the crash is reported; a crashed run is
  not replaced by a fresh seed.

## 5. What this run cannot settle

It is one space, two datasets, three seeds, and the discovered architectures are still not
trained to convergence. It tests whether the policy difference measured on the record survives
inside a live search; it does not test the NAS-Bench-201, NAS-Bench-301 or NAS-Bench-101
results of Secs. V–VI, which need no run of ours, and it does not make the operator-rewriting
budget result of Sec. IX less dependent on the acceptance rule for anything else.
