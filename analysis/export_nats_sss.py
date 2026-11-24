"""Export NATS-Bench SSS (hp=90, test accuracy) to data/external/nats-bench-sss.csv for s63.

Usage (needs the authors' `nats_bench` package and their "simple" archive, see
data/external/PROVENANCE_EXTRA.md):
    python3 export_nats_sss.py <path>/NATS-sss-v1_0-50262-simple

Reads the authors' 'simple' archive (one small file per architecture) in fast mode, one
architecture at a time, and drops each from the API's cache after use, so memory stays flat.
"""
import csv, sys
from nats_bench import create
src, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "../data/external/nats-bench-sss.csv")
api = create(src, 'sss', fast_mode=True, verbose=False)
n = len(api)
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh, lineterminator='\n')
    w.writerow(['arch', 'acc_cifar10', 'acc_cifar100', 'acc_ImageNet16-120'])
    for i in range(n):
        row = [api.meta_archs[i]]
        for ds in ('cifar10', 'cifar100', 'ImageNet16-120'):
            row.append(api.get_more_info(i, ds, hp='90', is_random=False)['test-accuracy'])
        w.writerow(row)
        api.arch2infos_dict.pop(i, None)
        if i % 4096 == 0:
            print('  %d / %d' % (i, n), flush=True)
print('exported', n, 'architectures to', out, flush=True)
