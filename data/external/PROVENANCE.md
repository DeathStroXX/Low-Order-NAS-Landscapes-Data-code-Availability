# Third-party benchmark tables

Neither file here is ours and neither is modified. Both are fetched unchanged from the
repository its authors publish it in, and both are byte-identical to what that repository
serves. They are kept in this directory, separate from `data/`, so that the line between the
data this study released and the data it merely reads stays visible.

If you prefer not to take them from this archive, delete the directory and run the two commands
below; every script that reads them checks for them and prints the command if they are absent.

## `nas-bench-macro_cifar10.json` (1.2 MB)

NAS-Bench-Macro, from *Prioritized Architecture Sampling with Monte-Carlo Tree Search*, Su
et al., CVPR 2021. Eight searchable layers, three candidate blocks each (identity, MB3_K3,
MB6_K5), so a complete `3^8 = 6,561` factorial; every architecture trained from scratch three
independent times on CIFAR-10. Each entry carries the three test accuracies, their mean and
standard deviation, the parameter count and the FLOPs.

```bash
curl -L -o data/external/nas-bench-macro_cifar10.json \
  https://raw.githubusercontent.com/xiusu/NAS-Bench-Macro/master/data/nas-bench-macro_cifar10.json
```

Read by `analysis/s57_benchmarks.py`. The script verifies the design is a complete `3^8`
factorial with one observation per cell and that `mean_acc` equals the mean of the three
per-seed accuracies before it decomposes anything.

## `HW-NAS-Bench-v1_0.pickle` (10.8 MB)

HW-NAS-Bench, from *HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark*, Li
et al., ICLR 2021. For every one of the 15,625 NAS-Bench-201 cells it reports latency on six
platforms and energy on three, plus Eyeriss's arithmetic intensity. Five of those columns are
measured by running the architecture on the device; four come from analytic cost models, and
`analysis/s58_hardware.py` never pools the two kinds, because they answer different questions
here.

```bash
curl -L -o data/external/HW-NAS-Bench-v1_0.pickle \
  https://raw.githubusercontent.com/GATECH-EIC/HW-NAS-Bench/main/HW-NAS-Bench-v1_0.pickle
```

**This file is a pickle from outside this project, so loading it naively would execute whatever
it contains.** `analysis/s58_hardware.py` reads it with a restricted unpickler that permits only
the NumPy array and container types the file legitimately needs and refuses anything else. If
you fetch it yourself, read it the same way, or inspect it first.

The script also verifies the alignment rather than assuming it: each row's `arch_str` is decoded
to a genotype and the design must come out a complete `5^6` factorial, covered exactly once.

## `nb201_all.pickle` (9.5 MB, kept in `data/`)

The NAS-Bench-201 accuracy table as NASLib redistributes it: all 15,625 cells with final
accuracy, parameters, FLOPs and measured latency on CIFAR-10, CIFAR-100 and ImageNet16-120.
Read by `analysis/s35_nb201.py` and every NAS-Bench-201 step after it.

```bash
curl -L -o data/nb201_all.pickle \
  https://raw.githubusercontent.com/automl/NASLib/Develop/naslib/data/nb201_all.pickle
```

## Checksums of the copies this study read

Fetched with the commands above on 22 November 2026 and byte-identical to the copies the
analysis used:

```
b93135dacdf16327733f0c165ee9f431f624e1f896b56b4abdc24df99e5be45e  data/nb201_all.pickle
686dee8363cc21cca8c1f321ad5749454f2d9b9ca5972682e137e539bc522297  data/external/nas-bench-macro_cifar10.json
57f92c2d7b88814fa337d00f4597c8ca30d507c5a855c18d6fac57e5ad872700  data/external/HW-NAS-Bench-v1_0.pickle
```

If an upstream file changes, `sha256sum` against these lines says so before any number moves.

## What is deliberately not here

The zero-cost proxy scores of the scope section are **NAS-Bench-Suite-Zero** (Krishnakumar
et al., NeurIPS 2022 Datasets and Benchmarks), which we do not redistribute. Fetch them from a
NASLib checkout:

```bash
git clone https://github.com/automl/NASLib && cd NASLib
bash scripts/zc/bash_scripts/download_nbs_zero.sh nb201   # also: nb101, nb301
# then move naslib/data/zc_nasbench*.json into this project's data/
```

You do not need them to check the paper: every number the manuscript quotes from those scores
is already in `analysis/results/zc_nb201.json`, `zc_spaces.json` and `encoding.json`.

## Licences

Each table is used under the licence of the repository that publishes it; consult those
repositories, which are linked above. We claim nothing over either file and redistribute them
here only so that the analysis in this archive can be rerun without a network.
