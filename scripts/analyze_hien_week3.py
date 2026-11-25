"""Audit saved validation runs, update the 18-run matrix, and prepare paired analysis.

Optional --run arguments enable checkpoint-backed error and attention exports.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10

from scripts.summarize import load_runs, summarize, save_summary_csv
from src.data import load_manifest, manifest_digest, make_eval_transform
from src.train import factory
from src.visualize import inspect_run, save_figure


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed_positions(manifest, targets):
    """First two original CIFAR indices per class, independent of model predictions."""
    positions = {index: position for position, index in enumerate(manifest['val_indices'])}
    selected = []
    for label in range(10):
        indices = sorted(i for i in positions if targets[i] == label)[:2]
        selected.extend(positions[i] for i in indices)
    return selected


def paired_errors(vit, resnet, labels, indices):
    if not (len(vit) == len(resnet) == len(labels) == len(indices)) or len(set(indices)) != len(indices):
        raise ValueError('Paired predictions must have equal length and unique indices')
    rows = []
    for index, label, v, r in zip(indices, labels, vit, resnet):
        group = ('both_correct' if v == label and r == label else
                 'vit_only_wrong' if v != label and r == label else
                 'resnet_only_wrong' if v == label and r != label else 'both_wrong')
        rows.append(dict(index=index, label=label, vit=v, resnet=r, group=group))
    return rows


def audit(rows, log_root):
    """Reject duplicate seeds and mismatched stored evidence before aggregation."""
    seen = set()
    for row in rows:
        model = row['model'].split(':')[0].split('.')[-1]
        ratio = f"{float(row['ratio']):.2f}"
        pair = (int(row['split_seed']), int(row['train_seed']))
        key = model, ratio, *pair
        if key in seen or model not in ('vit', 'resnet') or ratio not in ('0.10', '0.50', '1.00'):
            raise ValueError(f'Duplicate or unexpected experiment {key}')
        if pair not in ((2026, 42), (2027, 43), (2028, 44)) or row['split'] != 'val':
            raise ValueError(f'Unexpected split or seed {key}')
        seen.add(key)
        manifest = load_manifest(pair[0])
        if row['manifest_sha256'] != manifest_digest(manifest):
            raise ValueError(f'Manifest differs for {key}')
        reference = yaml.safe_load(Path(f'configs/week2/{model}_{round(float(ratio)*100)}.yaml').read_text())
        reference['seed'], reference['data']['split_seed'] = pair[1], pair[0]
        config_path = Path(row['config_path'])
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            if config != reference:
                raise ValueError(f'Configuration differs from locked recipe for {key}')
        if int(row['params']) != {'vit': 274282, 'resnet': 272474}[model]:
            raise ValueError(f'Parameter count differs for {key}')
        log = log_root / f'{model}_{round(float(ratio)*100)}_metrics.csv'
        if pair != (2026, 42):
            log = Path(row['checkpoint_path']).parent / 'metrics.csv'
        history = load_runs(log)
        best = max(history, key=lambda h: float(h['val_accuracy']))
        steps = 100 * math.ceil(len(manifest['subsets'][ratio]) / 128)
        if ([int(h['epoch']) for h in history] != list(range(1, 101))
                or int(row['global_step']) != steps or int(history[-1]['global_step']) != steps
                or int(row['best_epoch']) != int(best['epoch'])
                or int(row['total']) != 5000
                or abs(int(row['correct']) / 5000 - float(row['accuracy'])) > 1e-12
                or abs(float(row['accuracy']) - float(best['val_accuracy'])) > 1e-12):
            raise ValueError(f'Log/checkpoint record mismatch for {key}')
        total_seconds = sum(float(h['train_seconds']) for h in history)
        if not math.isclose(total_seconds, float(row['train_seconds_total']), rel_tol=1e-6):
            raise ValueError(f'Training time differs for {key}')
        row['architecture'] = model
        row['config_available'] = config_path.exists()
        row['train_accuracy_at_best'] = float(best['train_accuracy'])
        row['last_val_accuracy'] = float(history[-1]['val_accuracy'])
    return rows


def export_pairs(directories, output):
    inspected = {}
    for directory in directories:
        config, result, row, _ = inspect_run(directory)
        key = row['ratio'], config['data']['split_seed'], config['seed']
        model = row['architecture']
        if model in inspected.setdefault(key, {}):
            raise ValueError('Duplicate model in checkpoint pair')
        inspected[key][model] = directory, config, result, row
    for (ratio, split_seed, train_seed), models in inspected.items():
        if set(models) != {'vit', 'resnet'}:
            raise ValueError('Provide both ViT and ResNet for each ratio and seed')
        vd, vc, vr, vrow = models['vit']
        _, _, rr, rrow = models['resnet']
        if vr['labels'] != rr['labels'] or vrow['manifest_sha256'] != rrow['manifest_sha256']:
            raise ValueError('Prediction labels or manifests differ')
        manifest = load_manifest(split_seed)
        raw = CIFAR10(vc['data']['root'], train=True, download=False)
        indices = manifest['val_indices']
        if vr['labels'] != [raw.targets[i] for i in indices]:
            raise ValueError('Prediction order differs from validation manifest')
        name = f'd{round(float(ratio)*100)}_s{train_seed}'
        errors = paired_errors(vr['predictions'], rr['predictions'], vr['labels'], indices)
        write_csv(output / f'{name}_errors.csv', errors, list(errors[0]))
        counts = {group: sum(e['group'] == group for e in errors)
                  for group in ('both_correct', 'vit_only_wrong', 'resnet_only_wrong', 'both_wrong')}
        (output / f'{name}_error_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
        for architecture, result in (('vit', vr), ('resnet', rr)):
            confusion = np.zeros((10, 10), dtype=int)
            np.add.at(confusion, (result['labels'], result['predictions']), 1)
            np.savetxt(output / f'{name}_{architecture}_confusion.csv', confusion, delimiter=',', fmt='%d')
        examples = []
        for group in ('both_correct', 'vit_only_wrong', 'resnet_only_wrong', 'both_wrong'):
            examples.extend(sorted((e for e in errors if e['group'] == group), key=lambda e: e['index'])[:2])
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for axis in axes.flat:
            axis.axis('off')
        for axis, example in zip(axes.flat, examples):
            axis.imshow(raw[example['index']][0])
            axis.set_title(f"{example['group']}\nid={example['index']} true={example['label']}\nViT={example['vit']} CNN={example['resnet']}", fontsize=8)
        save_figure(fig, output / f'{name}_errors.png')
        selected = fixed_positions(manifest, raw.targets)
        model = factory(vc['model']['factory'])(vc)
        model.load_state_dict(torch.load(vd / 'best.pt', weights_only=True, map_location='cpu')['model'])
        device = vrow['evaluation_device']
        model.to(device).eval()
        fig, axes = plt.subplots(10, 2, figsize=(6, 24))
        transform = make_eval_transform()
        for axis, position in zip(axes.flat, selected):
            index = indices[position]
            original, label = raw[index]
            with torch.inference_mode():
                logits = model(transform(original).unsqueeze(0).to(device))
                if int(logits.argmax(1)) != vr['predictions'][position]:
                    raise ValueError('Attention prediction differs from evaluator')
                weights = model.get_attention_weights()[-1][0, :, 0, 1:].mean(0)
            if weights.numel() != 64:
                raise ValueError('Expected 8x8 patch grid')
            heat = weights.reshape(8, 8).repeat_interleave(4, 0).repeat_interleave(4, 1).cpu().numpy()
            axis.imshow(original)
            axis.imshow(heat, cmap='magma', alpha=0.48, interpolation='nearest')
            axis.set_title(f'id={index} true={label} pred={vr["predictions"][position]}')
            axis.axis('off')
        save_figure(fig, output / f'{name}_attention_fixed.png')
        provenance = dict(vit=vrow, resnet=rrow, selected_indices=[indices[p] for p in selected],
                          layer='last', heads='mean', query='CLS', key='patches', grid=[8, 8])
        (output / f'{name}_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('results/runs.csv'))
    parser.add_argument('--logs', type=Path, default=Path('results/team_week2'))
    parser.add_argument('--out', type=Path, default=Path('results/hien_week3'))
    parser.add_argument('--run', type=Path, action='append', default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = audit(load_runs(args.csv), args.logs)
    args.out.mkdir(parents=True, exist_ok=True)
    save_summary_csv(summarize(rows), args.out / 'summary.csv')
    fields = ('architecture', 'ratio', 'split_seed', 'train_seed', 'accuracy', 'params',
              'global_step', 'train_seconds_total', 'best_epoch', 'train_accuracy_at_best',
              'last_val_accuracy', 'checkpoint_path')
    write_csv(args.out / 'per_seed.csv', [{k: r[k] for k in fields} for r in rows], fields)
    expected = load_runs(Path('results/hien_week2/run_matrix.csv'))
    for entry in expected:
        match = [r for r in rows if all(r[k] == entry[k] for k in ('architecture', 'ratio', 'split_seed', 'train_seed'))]
        entry['status'] = ('records_verified_checkpoint_missing' if match and not Path(match[0]['checkpoint_path']).exists()
                           else 'records_verified_checkpoint_present' if match else 'awaiting_handoff')
    write_csv(args.out / 'run_matrix.csv', expected, list(expected[0]))
    fig, axis = plt.subplots()
    for architecture in ('vit', 'resnet'):
        group = sorted((r for r in rows if r['architecture'] == architecture), key=lambda r: float(r['ratio']))
        for pair in sorted({(r['split_seed'], r['train_seed']) for r in group}):
            run_group = [r for r in group if (r['split_seed'], r['train_seed']) == pair]
            axis.plot([float(r['ratio']) * 100 for r in run_group],
                      [float(r['accuracy']) * 100 for r in run_group], 'o-',
                      label=f'{architecture} split={pair[0]}, seed={pair[1]}')
    axis.set(xlabel='Training data (%)', ylabel='Validation accuracy (%)', xticks=[10, 50, 100])
    axis.legend()
    save_figure(fig, args.out / 'validation_scaling.png')
    raw = CIFAR10('data', train=True, download=False)
    selected = []
    for seed in (2026, 2027, 2028):
        m = load_manifest(seed)
        selected.extend(dict(split_seed=seed, position=p, index=m['val_indices'][p],
                             label=raw.targets[m['val_indices'][p]]) for p in fixed_positions(m, raw.targets))
    write_csv(args.out / 'attention_selection.csv', selected, list(selected[0]))
    report = dict(expected=18, records_verified=len(rows), missing=18-len(rows),
                  checkpoint_present=sum(Path(r['checkpoint_path']).exists() for r in rows),
                  config_available=sum(r['config_available'] for r in rows),
                  source_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                  paired_analysis='pending checkpoints' if not args.run else 'exported')
    if args.run:
        export_pairs(args.run, args.out)
    (args.out / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
