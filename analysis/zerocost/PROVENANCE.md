# `analysis/zerocost/` — vendored reference implementations

Nothing in this directory is ours. It is the reference implementation of the zero-cost
proxies, vendored so that the comparison in Section V of the paper is reproducible from this
archive alone, and so that no reader has to take our word for how a proxy was computed.

## Where it came from

| Path | Upstream | Licence |
|---|---|---|
| `measures/*.py` | `naslib/predictors/utils/pruners/measures/` of [automl/NASLib](https://github.com/automl/NASLib), itself derived from [SamsungLabs/zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas) (Abdelfattah et al., ICLR 2021) | Apache-2.0 |
| `predictive.py`, `p_utils.py`, `weight_initializers.py`, `__init__.py` | `naslib/predictors/utils/pruners/` | Apache-2.0 |
| `models/nasbench2.py`, `models/nasbench2_ops.py` | `naslib/predictors/utils/models/` — the NAS-Bench-201 cell network, built from the benchmark's own architecture string | Apache-2.0 |

Cloned from `https://github.com/automl/NASLib` at commit recorded in
`../results/zc_environment.json`. The Apache-2.0 headers are intact in every file.

## The two local modifications, both recorded in the files themselves

1. `predictive.py`: the upstream module imports `get_model_stats` (which requires
   `tensorwatch`) at import time. We never request the `flops` or `params` measures through
   this module — both are read exactly from the NAS-Bench-201 table — so the import is made
   lazy. No measure's arithmetic is touched.
2. `models/__init__.py`: added, because `p_utils.py` does `from ..models import *` and the
   upstream package `__init__` is not vendored. It re-exports the NAS-Bench-201 builder only.

`measures/model_stats.py` is vendored unchanged for completeness but is never called.

## What we supply ourselves

Only the data loader and the driver: `../s43_zc_compute.py`. It feeds one minibatch of real
CIFAR-10 training images to the measures through the upstream `find_measures` entry point,
with upstream's own defaults (`dataload='random'`, one batch; `dataload='grasp'` with one
image per class for `grasp`, as upstream does). Every proxy value in the paper is whatever
that function returns.
