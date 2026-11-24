# Papper2 — updated-log rebuild checkpoint

This is an intermediate data rebuild, **not the completed paper revision or a final Zenodo deposit**. The original manuscript, figures, tables, and uploaded archive have not been overwritten. No ZIP was created.

## What has been rebuilt

The rebuild reads only the updated JSONL files and matching terminal logs supplied in `Paper 2(2).zip`. It does not read the legacy gene CSV or use its old accuracy values.

| Quantity | Verified result |
|---|---:|
| Runs | 34 |
| Recorded evaluation rows (one UID per run) | 10,082 |
| Distinct UID–dataset combinations | 8,427 |
| Complete three-epoch learning curves | 10,082 |
| Epoch-level learning-curve rows | 30,246 |
| Evaluation rows with chromosomes explicitly in their own run | 2,988 |
| Additional rows linked by exact full UID to an inline chromosome | 1,169 |
| Evaluation rows with all 13 gene fields unavailable | 5,925 |

The 10,082 evaluations are **architecture–run observations**, not 10,082 distinct architectures. The analyses describe recorded evaluations; unsuccessful attempts absent from the evaluation JSONL files are not represented as successful observations.

## Files and use

| File | Purpose |
|---|---|
| `rebuild_updated_logs.py` | Standalone Python standard-library reconstruction script. |
| `evaluations_rebuilt_partial_genes.csv` | All updated observations, including first-, second-, and third-epoch accuracy, predicted fitness, run/seed, available genes, and provenance. |
| `learning_curves_rebuilt.csv` | Complete training/test curves; compatible columns with the original learning-curve export. |
| `rebuild_audit.json` | Per-run coverage and aggregate counts. |
| `source_manifest.json` | Relative paths, sizes, and SHA-256 digests for the 102 evaluation JSONL files and 34 pod logs used. |
| `validation.json` | Checks performed and output from the three rerun analysis scripts. |
| `common_updated_log_layout.py` | Original shared loader with the pod-log lookup adapted to the new separate `logs/pod_logs/` tree. This is not yet a fully revised gene loader. |
| `fitness_recomputed.json`, `proxy_recomputed.json`, `curves_recomputed.json` | Results from rerunning the corresponding original analyses against the updated logs. |
| `tab_*_recomputed.tex` | Regenerated fitness-provenance, proxy-fidelity, and learning-curve tables. Not yet integrated into the manuscript. |

Rebuild from an extracted original project:

```bash
python3 rebuild_updated_logs.py \
  --project /path/to/A_landscape_spectra_submission_v2 \
  --out /path/to/rebuilt_data
```

To repeat the three original analyses in a separate working copy, use `common_updated_log_layout.py` as `analysis/common.py`, then run `s10_fitness_provenance.py`, `s11_proxy_fidelity.py`, and `s14_learning_curves.py` from that copy's `analysis/` directory. Keep the original project unchanged. These scripts write their normal result and table filenames.

## Changes to the extraction

1. Read terminal logs from `logs/pod_logs/<method>/<dataset>/<run>/`; the original loader searched beside the evaluation JSONL files and therefore missed the new terminal-log location.
2. Join the three evaluation streams by **full UID within the same run**. Require matching UID sets and reject duplicate UIDs that would need a separate occurrence policy.
3. Recover all three epochs from the terminal logs, and validate the first and third test accuracies against the corresponding JSONL values within 0.011 percentage points. Keep separate training attempts and reject ambiguous curve matches.
4. Recover seed from the run directory and generation/iteration from the log context where available. Preserve run identity rather than assigning a shared UID to the first run that contains it.
5. Keep explicit gene provenance. `inline_in_same_run` means the chromosome was recorded alongside that evaluation; `exact_uid_match_to_inline_log` means it was joined from a supplied log with the same full UID. All repeated inline UID–chromosome associations were checked for agreement.
6. Leave all 13 gene fields blank for `not_present_in_supplied_logs`. Missing values are not inferred from the old CSV, replaced with zeros, or silently mapped to the first gene category.

The independent rebuilt curves match all 30,246 rows produced by the original learning-curve parser after repairing its log path.

## Preliminary changes in measured results

These values use the original analysis definition: mean within-run Spearman correlation with three-epoch accuracy, with 17 runs per dataset. They are descriptive reruns, not newly established statistical differences between old and updated experiments.

| Signal | Dataset | Old reported mean rho | Updated mean rho |
|---|---|---:|---:|
| One-epoch accuracy | CIFAR-10 | 0.918 | 0.931 |
| One-epoch accuracy | CIFAR-100 | 0.881 | 0.896 |
| LLM accuracy predictor | CIFAR-10 | 0.405 | 0.567 |
| LLM accuracy predictor | CIFAR-100 | 0.376 | 0.326 |
| Two-epoch accuracy, recovered-curve analysis | CIFAR-10 | 0.977 | 0.983 |
| Two-epoch accuracy, recovered-curve analysis | CIFAR-100 | 0.975 | 0.982 |

The LLM-predictor result changes in different directions across the two datasets, so a blanket statement about its change would be misleading. Early-epoch accuracy remains more strongly rank-correlated with the three-epoch reference in these data. This reference does not validate extrapolated converged accuracy; that would require longer training outcomes.

The fitness-provenance script matches 10,086 printed fitness events to recorded observations; four are repeated print events, so this must not be reported as 10,086 distinct evaluations. All matched events identify the LLM predictor as their source. The script also counts 514 printed duplicate-penalty events.

## What is still needed before the full paper rewrite

The original generator of `gene_diversity_benchmark_updated.csv` was not found among the supplied Python or shell scripts. `analysis/common.py`, `s53_spectrum.py`, and `s5_predictor_validation.py` **read** it; they do not reconstruct its gene fields. `s14_learning_curves.py` reconstructs a different CSV containing learning curves.

The baseline and LLM evaluation JSONL records contain only `uid`, `accuracy`, `is_cached`, and `timestamp`. Frozen-control records additionally contain chromosomes. Baseline terminal logs contain accuracy/fitness records and references to saved models and statistics, but no full 13-gene records. The terminal logs explicitly refer to generated `architectures/` and `stats/` outputs; those referenced run-output directories are not present in the uploaded archive.

The original CSV generator may read those outputs or another source not included here. Obtain that generator and inspect its actual input paths before deciding which further files are required. A previously generated CSV need not be supplied if its generator and required inputs are available.

Do not use this partial-gene CSV as if it were the completed gene dataset. In particular, the original `surrogate.py` encoder falls back to the first category for an unrecognized gene value, so passing blank fields through it could produce plausible but invalid analyses. Surrogate, gene-interaction, transfer, and dependent policy claims remain pending until gene coverage is resolved and their analyses rerun.

The main manuscript, supplementary manuscript, final Overleaf folder, reviewer guide, and final change README remain to be completed after that dependency is resolved.
