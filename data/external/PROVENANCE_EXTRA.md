# Two further complete factorials: what they are, and where to get them

**This archive does not redistribute either of the two tables described below, in any form,
whole or in part.** `PROVENANCE.md` in this directory covers the third-party tables that *are*
shipped here (NAS-Bench-Macro and HW-NAS-Bench); this file covers the two that are not, and the
pipeline that reads them if you fetch them yourself.

`analysis/s63_extra_grids.py` reads both. If a table is absent it prints the fetch instruction
for it, skips that grid and exits without writing anything: no result file in this archive
contains a number taken from a table that was not read.

## Why these two

The decomposition in Section IV is exact for any space that is a complete factorial with one
observation per cell, and two public NAS benchmarks are one besides NAS-Bench-201 and
NAS-Bench-Macro:

| benchmark | grid | cells | responses |
|---|---|---|---|
| NATS-Bench, size search space (SSS) | `8^5` | 32,768 | CIFAR-10, CIFAR-100, ImageNet16-120 |
| TransNAS-Bench-101, micro (cell) space | `4^6` | 4,096 | seven task types |

Each extends the measurement in a direction the shipped benchmarks cannot. NATS-Bench SSS
varies *size* rather than operation: its five factors are ordinal channel widths, so it asks
whether the concentration survives when the factors are quantities rather than categories.
TransNAS-Bench-101 micro varies the *task type* rather than its difficulty, which is the only
way to separate a property of the search space from a property of the objective.

## NATS-Bench, size search space (SSS)

**What it is.** The size search space of NATS-Bench: five searchable layers, each taking one of
eight channel widths (8, 16, 24, 32, 40, 48, 56, 64), so a complete `8^5 = 32,768` factorial,
with every architecture trained and evaluated on CIFAR-10, CIFAR-100 and ImageNet16-120. The
*topology* search space of the same benchmark is NAS-Bench-201's `5^6` grid, which this paper
already analyses from `data/nb201_transfer.csv`; it is the size space that is new here.

**Who publishes it.** Dong et al., *NATS-Bench: Benchmarking NAS Algorithms for Architecture
Topology and Size*. The benchmark files are distributed by the authors' own repository:

    https://github.com/D-X-Y/NATS-Bench

**How to fetch it.** Take the current download link for the size-space benchmark file (the
files whose names begin `NATS-sss`) from that repository's README. We deliberately do not quote
a direct file URL: the authors have moved the hosting of these files more than once, and a
hard-coded link that has gone stale is worse than no link. Install the authors' `nats_bench`
package the same way, from that repository.

**Licence situation.** The table is used under the licence of the repository that publishes it;
read that repository's licence file and its README before redistributing anything derived from
it. We claim nothing over the table and ship no part of it.

**Export it for `s63`.** The released file is a NATS-Bench pickle, which this archive cannot and
will not load with numpy and pandas alone, so `s63` reads a plain export you make once from your
own copy. Write `data/external/nats-bench-sss.csv` (or point `NATS_SSS_PATH` at it) with one row
per architecture, the five channel widths in an `arch` column, and one accuracy column per
dataset:

```python
# illustrative; check the call signatures against the nats_bench version you installed
from nats_bench import create
api = create('<your NATS-sss file>', 'sss', fast_mode=True, verbose=False)
import csv
with open('data/external/nats-bench-sss.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['arch', 'acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'])
    for i, arch in enumerate(api.meta_archs):          # 32,768 of them
        row = [arch]
        for ds in ('cifar10', 'cifar100', 'ImageNet16-120'):
            info = api.get_more_info(i, ds, hp='90', is_random=False)
            row.append(info['test-accuracy'])
        w.writerow(row)
```

Use one accuracy definition throughout and say which one you used: `s63` records the file's
SHA-256 in its output, but it cannot know which split or budget you exported, and the number of
training epochs (`hp`) changes the response. Any consistent choice is a valid landscape; mixing
two is not.

**What this study used.** On 24 November 2026 we fetched the authors' release
`NATS-sss-v1_0-50262` from the links in their README (Google Drive, row dated 2020.08.31):
the single-file benchmark `NATS-sss-v1_0-50262.pickle.pbz2` (MD5
`fc568de484928516689f114542a50262`, matching the `50262` in its name) and the archive
`NATS-sss-v1_0-50262-simple.tar` (MD5 `208c64fad4ade8837ead108bb108c049`, 32,768 per-architecture
files). The single-file version needs more memory than a 32 GB machine has free, so we read the
archive with `analysis/export_nats_sss.py`, which uses the authors' `nats_bench` package in fast
mode: test accuracy after the benchmark's 90-epoch schedule (`hp='90'`) on CIFAR-10, CIFAR-100
and ImageNet16-120. The resulting export has SHA-256
`e622725410c03b814274d401426dbe7763fa20e5817246619a912db8476d4ca4`; it is not redistributed.

## TransNAS-Bench-101, micro (cell) space

**What it is.** TransNAS-Bench-101 evaluates each architecture on seven tasks of different
*types* drawn from Taskonomy -- object classification, scene classification, jigsaw, room
layout, surface normal, semantic segmentation and autoencoding -- rather than on one task at
three difficulties. Its micro (cell) space fixes the macro skeleton and searches a four-node
cell with six edges and four operations per edge, so a complete `4^6 = 4,096` factorial. The
benchmark's architecture names carry the cell as six digits after the skeleton, e.g.
`64-41414-1_02_333`. The companion *macro* space of the same benchmark is **not** a complete
factorial (its skeletons vary in length), which is why only the micro space appears here.

**Who publishes it.** Duan et al., *TransNAS-Bench-101: Improving Transferability and
Generalizability of Cross-Task Neural Architecture Search*, CVPR 2021. The benchmark file and
its API are distributed by the authors themselves.

**How to fetch it.** The authors serve the file on request, or after a short registration,
rather than as an open download, so there is no stable URL for this file to quote. Follow the
link given in the paper and on its project page, and accept the terms stated there. Those terms
are the reason this archive ships a pipeline rather than a result for this grid.

**Licence situation.** The table is made available for research use on the terms its authors
state at the point of download, which include terms inherited from Taskonomy, the source of the
tasks. Read them; they are the operative licence, not anything in this archive. We claim nothing
over the table and ship no part of it.

**Export it for `s63`.** The released file is a PyTorch checkpoint, which this archive does not
load, so again `s63` reads a plain export. Write
`data/external/transnas-bench-101-micro.csv` (or point `TRANSNAS_PATH` at it) with one row per
micro architecture, the architecture name in an `arch` column, and one column per task holding
that task's own primary metric. Note that the seven metrics are not comparable to one another
(top-1 accuracy, SSIM, mIoU, a negative loss), which is deliberate: each is decomposed on its
own and only the *shape of its spectrum* is compared across tasks, never its level. Use the
authors' API to read, for each of the 4,096 micro architectures and each task, that task's best
recorded value on one fixed split, and name the columns after the tasks.

## Expected input format

`analysis/s63_extra_grids.py` holds the authoritative statement of what it expects in its
`EXPECTED_FORMAT` string, and prints it when a file is present but cannot be read as described.
In summary, for either table:

* `.csv` (or `.tsv`): one header row, one row per architecture, no index column.
  * One design column: `arch`, holding the benchmark's own identifier -- the five channel widths
    separated by any non-digit for NATS-Bench SSS (`8:16:32:48:64`), or the six-operation cell
    code for TransNAS micro, either as the full name (`64-41414-1_02_333`) or as the bare
    digits (`102333`). Alternatively `f0..f4` (SSS) or `f0..f5` (TransNAS), holding 0-based
    level indices.
  * Every other column must be numeric, and each one is decomposed separately as a response
    under the name the header gives it.
* `.json`: one object mapping the architecture identifier to an object of response name to
  number.
* Exactly one row per cell: 32,768 rows for SSS, 4,096 for TransNAS micro. A missing,
  duplicated or out-of-range cell is refused, and so is a response column with a missing value.
  The decomposition is exact only on a complete design, so a partial table is a different
  measurement and the script will not quietly make it.

## What `s63` writes

Only if at least one grid was read:

* `analysis/results/extra_grids.json` -- per-order and cumulative variance shares of every
  response, the analytic white-noise null of that grid shape, the skewness and top-decile
  spread of every response, and the provenance of the file each came from (path, SHA-256, size,
  number of architectures, which design column was used, and whether the path came from an
  environment variable).
* `tables/tab_extra.tex` -- the same, in the style of `tables/tab_bench.tex`.

Both files are reproducible from your own copy of the tables and from nothing else in this
archive, which is the point of keeping the pipeline separate from the data.
