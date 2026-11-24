"""Check the per-architecture record against the run logs it describes.

data/gene_diversity_benchmark.csv holds one row per architecture each of the 34 runs
evaluated: its run, its identifier, its 1-epoch and 3-epoch accuracy, the predictor's value,
its 13 genes, and the generated model file the genes were read from. The JSONL logs in logs/
record the identifier, the accuracies and the predictor value of every evaluation, and the
frozen-control logs also record the chromosome. This script requires that

  1. the record and the logs list the same (run, identifier) pairs, no more and no fewer;
  2. every row's acc1, acc3 and pred equal the values its run's logs record;
  3. every row has all 13 genes, each a value of the search space;
  4. for every frozen-control row, the 13 genes equal the chromosome its log records;
  5. every row's identifier is the identifier in its model file's name.

It fails on the first violated condition and writes results/record_check.json otherwise.
"""
import csv, json, os, sys

import common

SPACE = {
    'n_columns': {'2', '3', '4'}, 'n_blocks': {'2', '3', '4'}, 'base_channels': {'16', '32', '64'},
    'dropout_prob': {'0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5'},
    'fc_dropout': {'0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5'},
    'lr': {'0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.008', '0.01'},
    'momentum': {'0.75', '0.8', '0.85', '0.88', '0.9', '0.92', '0.95', '0.98'},
    'kernel_size': {'1', '3', '5', '7'}, 'activation': {'ReLU', 'GELU', 'LeakyReLU', 'SiLU'},
    'pooling_type': {'Max', 'Avg'}, 'conv_type': {'Standard', 'Depthwise'},
    'norm_type': {'BatchNorm', 'InstanceNorm'}, 'optimizer_type': {'SGD', 'AdamW', 'RMSprop'},
}


def fail(msg):
    sys.exit('record check FAILED: ' + msg)


def main():
    logged = {}
    for ds in common.DATASETS:
        for m in common.METHODS:
            for r in common.runs(m, ds):
                for u, a1, a3, p, c in zip(r['uids'], r['acc1'], r['acc3'], r['pr'], r['chrom']):
                    logged[(f"{m}/{ds}/{r['dir']}", u)] = (float(a1), float(a3), float(p), c)
    rows = list(csv.DictReader(open(os.path.join(common.DATA, 'gene_diversity_benchmark.csv'))))
    keys = {(r['run'], r['uid']) for r in rows}
    if len(keys) != len(rows):
        fail('%d duplicate (run, uid) rows' % (len(rows) - len(keys)))
    if keys != set(logged):
        fail('%d pairs only in the record, %d only in the logs'
             % (len(keys - set(logged)), len(set(logged) - keys)))
    n_frozen = 0
    for r in rows:
        a1, a3, p, c = logged[(r['run'], r['uid'])]
        for col, v in (('acc1', a1), ('acc3', a3), ('pred', p)):
            if abs(float(r[col]) - v) > 1e-9:
                fail('%s of %s in %s is %s, the log says %s' % (col, r['uid'], r['run'], r[col], v))
        for g, ok in SPACE.items():
            if r[g] not in ok:
                fail('gene %s of %s is %r, not a value of the space' % (g, r['uid'], r[g]))
        if c:
            n_frozen += 1
            for g in SPACE:
                if float(r[g]) != float(c[g]) if g not in ('activation', 'pooling_type', 'conv_type', 'norm_type', 'optimizer_type') else r[g] != c[g]:
                    fail('gene %s of %s differs from its frozen log (%r vs %r)' % (g, r['uid'], r[g], c[g]))
        if os.path.basename(r['source_file']) != 'GenFractalNet-%s.py' % r['uid']:
            fail('model file %s does not carry the identifier %s' % (r['source_file'], r['uid']))
    out = dict(rows=len(rows), runs=len({r['run'] for r in rows}), pairs_match_logs=True,
               accuracies_match_logs=True, genes_in_space=True,
               frozen_rows_checked_against_logged_chromosome=n_frozen)
    json.dump(out, open(os.path.join(common.RES, 'record_check.json'), 'w'), indent=1)
    print('record check passed: %d rows, %d runs; accuracies and predictor values equal the logs; '
          'all genes in the space; %d frozen rows equal their logged chromosomes'
          % (out['rows'], out['runs'], n_frozen))


if __name__ == '__main__':
    main()
