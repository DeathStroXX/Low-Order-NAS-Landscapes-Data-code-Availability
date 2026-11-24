# Low-Order Structure in Neural Architecture Search Landscapes

**Exact spectra, a scale illusion, and budget policy — analysis code, data, logs and results**

[![DOI](https://zenodo.org/badge/DOI/ZENODO-DOI-PLACEHOLDER.svg)](https://doi.org/ZENODO-DOI-PLACEHOLDER)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

Two public NAS benchmarks tabulate every architecture of a complete factorial space, so their
accuracy functions can be decomposed *exactly* by interaction order, with nothing fitted. This
repository holds everything that turns those tables, and the logs of 34 genetic-algorithm search
runs over a 13-gene FractalNet space, into the numbers, tables and figures of the paper and its
supplement. Nothing here needs a GPU: the training runs are already recorded.

Authors: Dmitry Ignatov, Akashdeep Singh, Sai Revanth Mulukuledu, Radu Timofte
(Computer Vision Lab, CAIDAS & IFI, University of Würzburg). Submitted to IEEE Transactions on
Neural Networks and Learning Systems. Archived release: https://doi.org/ZENODO-DOI-PLACEHOLDER

---

## Quick start

```bash
git clone https://github.com/DeathStroXX/Low-Order-NAS-Landscapes-Data-code-Availability.git
cd Low-Order-NAS-Landscapes-Data-code-Availability
python3 -m venv .venv && . .venv/bin/activate
pip install -r analysis/requirements.txt

make sums                    # checksums of every file in the archive, and of the pod logs
python3 verify_results.py    # re-runs the analyses in a temporary copy and compares every
                             # result with the committed one; exit status 0 when all match
```

`verify_results.py` copies the archive to a temporary directory, runs the scripts there and
compares each result file value by value with the committed one under `analysis/results/`;
your checkout is not modified. It takes about five minutes; `--all` adds the long steps
(about 25 minutes more). Steps whose third-party table you have not fetched are reported as
skipped (see *Third-party tables* below), so the FractalNet results can be checked with no
download at all.

Four steps are not re-run by `verify_results.py`, and their committed results stand as
released: `s62_nb201_search.py` (the 500-seed NAS-Bench-201 search replay, hours of CPU) and
`s43`--`s45` (the zero-cost proxy comparisons, which need the NAS-Bench-Suite-Zero scores from
NASLib). `make reproduce` runs all of them.

---

## What is in this repository

```
README.md, LICENSE, CITATION.cff, SHA256SUMS
Makefile                    make sums | make reproduce
verify_results.py           re-run in a temporary copy and compare with the committed results
analysis/
├── common.py               reading the run logs and the per-architecture record
├── surrogate.py            the degree-2 ridge on genes and gene pairs
├── fractalbench.py         the surrogate benchmark of the FractalNet space and its GA harness
├── policies.py, statsutil.py
├── s02_ … s73_*.py         the analysis pipeline, one script per result (order: `make reproduce`)
├── zerocost/               reference implementations of the 13 zero-cost proxies (Apache-2.0)
├── confirm/                the pre-registration (unmodified), its runner and its scorer
├── rebuild/                an independent reconstruction of the record from the raw logs
├── requirements.txt        the package versions the committed results were made with
└── results/                one JSON per analysis step: every number in the paper is in one
data/
├── gene_diversity_benchmark.csv   the per-architecture record of the 34 runs (see below)
├── learning_curves.csv            1-, 2- and 3-epoch train/test accuracy of every evaluation
├── surrogate_predictions.csv      held-out surrogate predictions for every evaluation
├── nb201_transfer.csv             per-cell predictions on NAS-Bench-201
├── cost_model.csv                 parameter and multiply-accumulate counts over the grid
└── external/                      fetch commands and checksums for the third-party tables
logs/                       JSONL evaluation, predictor and evolution logs, 34 runs (130 files)
large_input_data/pod_logs/  the 34 terminal (pod) logs, unmodified, 164 MB, with SHA256SUMS
tables/, figures/           the tables and figures the manuscript includes, as generated
```

The manuscript sources are not part of this repository.

---

## Which output is which part of the manuscript

| Output | In the manuscript | Produced by |
|---|---|---|
| `tables/tab_spectrum.tex`, `figures/fig_spectrum_panels.pdf` | the exact spectra of the two benchmarks, and the recalibrated spectra | `s53_spectrum.py`, `s54_spectrum_tables.py`, `s55_figures.py` |
| `tables/tab_bench_main.tex`, `tables/tab_bench.tex` | NAS-Bench-Macro, the measured null and the HW-NAS-Bench noise floor (and, in the supplement, the exactness certificates) | `s57_benchmarks.py`, `s58_hardware.py`, `s60_tables.py` |
| `tables/tab_extra.tex` | the exact spectrum of NATS-Bench's size space (a third complete factorial) | `export_nats_sss.py` (from the authors' release), `s63_extra_grids.py` |
| `tables/tab_recal_full.tex` | recalibrate, transfer, or both, with standard errors | `s53_spectrum.py`, `s54_spectrum_tables.py` |
| `analysis/results/robustness.json`, `k_sensitivity.json` | equivalent cells, the rank-transform null, the scope test at small label counts; the regret comparisons at other shortlist sizes | `s74_robustness.py`, `s75_k_sensitivity.py` |
| `tables/tab_recal.tex` | recalibration against transfer, with the uncompressed FractalNet control | `s53_spectrum.py`, `s54_spectrum_tables.py` |
| `tables/tab_search.tex`, `tables/tab_search_sm.tex` | the replayed NAS-Bench-201 search against random search and regularized evolution | `s62_nb201_search.py` |
| `tables/tab_realization.tex`, `tables/tab_learned.tex` | every ranker against the exact degree-2 optimum; trained predictors at matched budgets | `s64_realization.py`, `s68_learned_predictors.py` |
| `figures/fig_body_panels_1col.pdf` | label efficiency, cost against fidelity, regret against shortlist size, what a prior is worth | `s69_body_panels.py` |
| `tables/tab_monotone.tex`, `tables/tab_targets.tex`, `tables/tab_bench.tex` | the monotone-map family; the other functions on the same grid; the benchmark comparison | `s65_monotone.py`, `s56_targets.py`, `s60_tables.py` |
| `tables/tab_fitness.tex`, `tables/tab_censoring.tex` | what the search selected on; the accuracy-floor truncation per configuration | `s10_fitness_provenance.py`, `s72_censoring.py` |
| `tables/tab_halving.tex`, `tables/tab_recipe.tex`, `tables/tab_sitl2.tex`, `tables/tab_transfer2.tex` | the budget policies on the FractalNet record | `s17_halving.py`, `s30_recipe.py`, `s31_sitl.py`, `s29_warmstart.py`, `s25_transfer.py`, `s71_merged_tables.py` |
| `tables/tab_zc_sm.tex`, `tables/tab_zc_spaces.tex`, `tables/tab_encoding.tex`, `figures/fig_zc.pdf` | the zero-cost proxy family, NAS-Bench-101 and -301, the encoding boundary | `s43_zc_proxies.py`, `s44_zc_spaces.py`, `s45_encoding.py` |
| `tables/tab_novelty.tex`, `figures/fig_decision.pdf` | the claim ledger and the procedure as a diagram | `s49_novelty.py`, `s50_decision.py` |
| remaining `figures/*.pdf` | supplementary figures | the script named in each caption |

---

## Full rebuild

`make reproduce` reruns every analysis script from the logs, the record and the third-party
tables, in dependency order, and rewrites `analysis/results/`, `tables/` and `figures/`. The
replayed NAS-Bench-201 search (`s62_nb201_search.py`, 500 seeds) takes hours; everything else
together takes under an hour on one CPU core.

---

## Data sources

| Input | Location | Used for |
|---|---|---|
| Logs of the 34 search runs | `logs/` (JSONL) and `large_input_data/pod_logs/` (terminal) | every FractalNet result; learning curves; what the search selected on |
| Per-architecture record | `data/gene_diversity_benchmark.csv` | the genes of every evaluation (see below) |
| NAS-Bench-201, NAS-Bench-Macro, HW-NAS-Bench, NATS-Bench (size space) | fetched, not redistributed: `data/external/PROVENANCE.md` gives each command and the SHA-256 of the copy this study read | the exact spectra and everything on those grids |
| NAS-Bench-Suite-Zero proxy scores | fetched from NASLib, not redistributed (`data/external/PROVENANCE.md`) | the zero-cost proxy comparison |
| Pre-registration | `analysis/confirm/PREREGISTRATION.md`, unmodified | the confirming run still outstanding |

**The per-architecture record.** One row per architecture each of the 34 runs evaluated
(10,082): its run, its 1- and 3-epoch accuracy, the predictor's value, its 13 genes, and the
generated model file the genes were read from, with that file's SHA-256. Only the frozen-control
logs print chromosomes, so the record is where the baseline and LLM-arm genes come from.
`analysis/s02_check_record.py` verifies it against the logs (same architectures in the same
runs, identical accuracies and predictor values, every gene a value of the space, and identical
genes for the 2,988 frozen rows whose logs record them); `verify_results.py` runs it first.

---

## Reproducibility

**What is guaranteed.** Every random quantity is seeded, so a rerun reproduces the committed
results for the package versions in `analysis/requirements.txt`; `verify_results.py` checks it.
The one exception is the neural predictor of `s68_learned_predictors.py`, whose results can move
in the seventh decimal on other hardware; nothing quoted in the paper is affected.

**Environment used for the committed results.** Python 3.10.6, numpy 1.26.4, scipy 1.15.3,
pandas 2.3.2, matplotlib 3.10.8, scikit-learn 1.7.2.

---

## Scope of the evidence

The exact spectral results hold on the two enumerated benchmarks (NAS-Bench-201, NAS-Bench-Macro)
and nowhere else by construction. The FractalNet record is used as an uncompressed control and
for the budget policies:

| Arm | CIFAR-10 runs | CIFAR-100 runs |
|---|---|---|
| Baseline GA (one 26-generation search) | 3 | 3 |
| Restart schedule, frozen handcrafted operators | 5 | 5 |
| Restart schedule, LLM-written operators (Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Mistral-7B) | 9 | 9 |

10,082 evaluations (8,427 distinct architectures), each trained for one and three epochs. The
search discarded candidates at an accuracy floor, so the record is left-truncated (6.8–16.0% of
attempts on CIFAR-10, 20.7–25.5% on CIFAR-100; `s72_censoring.py`). Budget-policy results on it
are differences between policies on one landscape, measured against the 3-epoch accuracy the
runs recorded, not improvements in fully trained accuracy. The one confirming training run is
pre-registered and not yet run.

---

## Citation

```bibtex
@misc{ignatov_loworder_nas_artifacts,
  author = {Ignatov, Dmitry and Singh, Akashdeep and Mulukuledu, Sai Revanth and Timofte, Radu},
  title  = {Low-Order Structure in Neural Architecture Search Landscapes: code, data and logs},
  year   = {2026},
  doi    = {ZENODO-DOI-PLACEHOLDER},
  url    = {https://github.com/DeathStroXX/Low-Order-NAS-Landscapes-Data-code-Availability}
}
```

Please also cite the article once it appears (PAPER-DOI-PLACEHOLDER).

---

## Licence

Code (`analysis/`, `Makefile`, `PROTECT.sh`, `verify_results.py`): MIT. Data, logs, results,
tables and figures: CC BY 4.0. See `LICENSE`. Vendored third-party code keeps its own licence.
