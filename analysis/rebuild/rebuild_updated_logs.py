#!/usr/bin/env python3
"""Rebuild observations from the supplied JSONL and pod logs without legacy CSV data.

Run: python3 rebuild_updated_logs.py --project PATH --out OUTPUT_DIRECTORY
Blank gene fields mean unavailable, never zero or the first search-space category.
This is an observation rebuild, not a replacement for the missing original gene extractor.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

GENES = {
    'LR': 'lr', 'Momentum': 'momentum', 'Activation': 'activation',
    'Kernel': 'kernel_size', 'Pooling': 'pooling_type', 'Conv_Type': 'conv_type',
    'Norm_Type': 'norm_type', 'Optimizer': 'optimizer_type',
    'FC_Dropout': 'fc_dropout', 'Dropout_Prob': 'dropout_prob',
    'N_Columns': 'n_columns', 'Base_Channels': 'base_channels', 'N_Blocks': 'n_blocks',
}
ARCH = re.compile(r'Evaluating unique arch \(checksum:\s*([0-9a-f]+)')
GEN = re.compile(r'Generation:\s*(\d+)\s*/\s*(\d+)')
ITER = re.compile(r'=== Iteration (\d+)/\d+')
EPOCH = re.compile(r'Train Acc:\s*([0-9.]+),\s*Test Acc:\s*([0-9.]+)')


def read_log(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    by_uid = {row['uid']: row for row in rows}
    if len(rows) != len(by_uid):
        raise ValueError(f'Duplicate UID needs an explicit occurrence policy: {path}')
    return by_uid


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pod_blocks(path):
    """Keep separate attempts; choose a completed attempt only after endpoint validation."""
    blocks = collections.defaultdict(list)
    generation = iteration = None
    uid = None
    buf = []
    context = None

    def close():
        if uid is not None:
            blocks[uid].append({'epochs': list(buf), **context})

    with path.open(errors='replace') as handle:
        for line_number, line in enumerate(handle, 1):
            match = ITER.search(line)
            if match:
                iteration = int(match.group(1))
            match = GEN.search(line)
            if match:
                generation = int(match.group(1))
            match = ARCH.search(line)
            if match:
                close()
                uid, buf = match.group(1), []
                context = dict(generation=generation, meta_iteration=iteration,
                               pod_start_line=line_number)
                continue
            match = EPOCH.search(line)
            if match and uid is not None:
                buf.append((100 * float(match.group(1)), 100 * float(match.group(2))))
    close()
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project, out = args.project.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs, gene_map = [], {}
    for true_file in sorted((project / 'logs').rglob('*_true.jsonl')):
        method, dataset, run = true_file.relative_to(project / 'logs').parts[:3]
        base = str(true_file)[:-len('_true.jsonl')]
        e1_file, pred_file = Path(base + '_1_epoch.jsonl'), Path(base + '.jsonl')
        e1, e3, pred = (read_log(f) for f in [e1_file, true_file, pred_file])
        if e1.keys() != e3.keys() or e3.keys() != pred.keys():
            raise ValueError(f'UID mismatch between the three logs: {run}')
        for uid, rec in e3.items():
            chrom = rec.get('chromosome')
            if chrom:
                genes = {g: chrom[g] for g in GENES.values()}
                if uid in gene_map and gene_map[uid][0] != genes:
                    raise ValueError(f'Conflicting chromosome for UID {uid}')
                gene_map.setdefault(uid, (genes, str(true_file.relative_to(project))))
        runs.append((method, dataset, run, e1, e3, pred, true_file))

    observations, curves, summaries, inputs = [], [], [], []
    for method, dataset, run, e1, e3, pred, true_file in runs:
        pod_files = sorted((project / 'logs/pod_logs' / method / dataset / run).glob('*.log'))
        if len(pod_files) != 1:
            raise ValueError(f'Expected exactly one pod log for {method}/{dataset}/{run}')
        pod = pod_files[0]
        blocks = pod_blocks(pod)
        prefix_counts = collections.Counter(uid[:8] for uid in e3)
        if any(count != 1 for count in prefix_counts.values()):
            raise ValueError(f'Ambiguous 8-character UID prefix in {run}')
        gene_counts, curve_counts = collections.Counter(), collections.Counter()
        for uid, rec in e3.items():
            a1, a3 = e1[uid]['accuracy'], rec['accuracy']
            candidates = blocks.get(uid[:8], [])
            good = [b for b in candidates if len(b['epochs']) == 3
                    and abs(b['epochs'][0][1] - a1) <= 0.011
                    and abs(b['epochs'][2][1] - a3) <= 0.011]
            # Avoid choosing arbitrarily when multiple different curves match the endpoints.
            distinct = {tuple(b['epochs']) for b in good}
            selected = good[0] if len(distinct) == 1 else None
            curve_status = ('verified_3_epochs' if selected else
                            'ambiguous' if len(distinct) > 1 else 'not_recovered')
            curve_counts[curve_status] += 1
            if rec.get('chromosome'):
                genes = {g: rec['chromosome'][g] for g in GENES.values()}
                gene_status = 'inline_in_same_run'
                gene_source = str(true_file.relative_to(project))
            elif uid in gene_map:
                genes, gene_source = gene_map[uid]
                gene_status = 'exact_uid_match_to_inline_log'
            else:
                genes, gene_source = {}, ''
                gene_status = 'not_present_in_supplied_logs'
            gene_counts[gene_status] += 1
            context = selected or (candidates[0] if len(candidates) == 1 else {})
            iteration = rec.get('meta_iteration', context.get('meta_iteration'))
            seed_match = re.match(r'seed(\d+)_', run)
            row = dict(uid=uid, dataset=dataset, pipeline=method,
                       run=f'{method}/{dataset}/{run}',
                       seed=int(seed_match.group(1)), meta_iteration=iteration,
                       generation=context.get('generation'), is_cached=rec['is_cached'],
                       status='recorded_evaluation', acc_epoch1=a1,
                       acc_epoch2=selected['epochs'][1][1] if selected else None,
                       acc_epoch3_true=a3, predicted_fitness=pred[uid]['accuracy'])
            row.update({col: genes.get(gene) for col, gene in GENES.items()})
            row.update(gene_status=gene_status, gene_source=gene_source,
                       curve_status=curve_status, timestamp=rec['timestamp'],
                       source_true_log=str(true_file.relative_to(project)),
                       source_pod_log=str(pod.relative_to(project)),
                       pod_start_line=context.get('pod_start_line'))
            observations.append(row)
            if selected:
                for epoch, (train_acc, test_acc) in enumerate(selected['epochs'], 1):
                    curves.append(dict(uid=uid, dataset=dataset, method=method,
                                       arm=method if method in ['frozen', 'baseline'] else 'llm',
                                       run=row['run'], epoch=epoch,
                                       train_acc=round(train_acc, 4), test_acc=round(test_acc, 4),
                                       acc_epoch3_true=a3, predicted_fitness=pred[uid]['accuracy']))
        summaries.append(dict(run=f'{method}/{dataset}/{run}', n_observations=len(e3),
                              genes=dict(gene_counts), curves=dict(curve_counts)))
        for f in [true_file, Path(str(true_file).replace('_true.jsonl', '_1_epoch.jsonl')),
                  Path(str(true_file).replace('_true.jsonl', '.jsonl')), pod]:
            inputs.append(dict(path=str(f.relative_to(project)),
                               sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                               size_bytes=f.stat().st_size))
    write_csv(out / 'evaluations_rebuilt_partial_genes.csv', observations)
    write_csv(out / 'learning_curves_rebuilt.csv', curves)
    report = dict(n_runs=len(runs), n_observations=len(observations),
                  n_uid_dataset_pairs=len({(r['uid'],r['dataset']) for r in observations}),
                  gene_coverage=dict(collections.Counter(r['gene_status'] for r in observations)),
                  curve_coverage=dict(collections.Counter(r['curve_status'] for r in observations)),
                  runs=summaries,
                  note='No values were read from the old CSV. Unknown gene fields are blank.')
    (out / 'rebuild_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'source_manifest.json').write_text(json.dumps(inputs, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
