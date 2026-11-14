"""Validation-only figures and run checks for the week-two handoff."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10

from src.data import build_loaders, load_manifest
from src.evaluate import build_run_row, evaluate_checkpoint
from src.train import factory


def inspect_run(directory):
    config_path = directory / 'config.yaml'
    checkpoint = directory / 'best.pt'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    result = evaluate_checkpoint(checkpoint, config, 'val')
    row = build_run_row(config_path, checkpoint, result, 'val')
    with (directory / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        history = list(csv.DictReader(stream))
    best = max(history, key=lambda item: float(item['val_accuracy']))
    if (int(best['epoch']) != row['best_epoch']
            or int(best['epoch']) != row['checkpoint_epoch']
            or abs(float(best['val_accuracy']) - result['accuracy']) > 1e-7):
        raise ValueError(f'{directory}: validation log does not match best checkpoint')
    required = ('commit', 'params', 'manifest_sha256', 'checkpoint_sha256',
                'protocol_id', 'global_step', 'train_seconds_total')
    if any(row.get(key) in (None, '') for key in required):
        raise ValueError(f'{directory}: missing run metadata')
    data = config['data']
    model = config['model']['factory'].split(':')[0].split('.')[-1]
    manifest = load_manifest(data['split_seed'], data.get('splits_dir', 'results/splits'))
    row.update(architecture=model, train_images=len(manifest['subsets'][row['ratio']]),
               metrics_path=(directory / 'metrics.csv').as_posix())
    return config, result, row, history


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def attention_examples(directory, config, result, output):
    """Use the first correct and incorrect images in the fixed manifest order."""
    _, loader = build_loaders(config)
    dataset = loader.dataset
    raw = CIFAR10(config['data']['root'], train=True, download=False)
    model = factory(config['model']['factory'])(config)
    state = torch.load(directory / 'best.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()
    positions = {}
    for i, (label, prediction) in enumerate(zip(result['labels'], result['predictions'])):
        positions.setdefault(label == prediction, i)
        if len(positions) == 2:
            break
    fig, axes = plt.subplots(len(positions), 2, squeeze=False, figsize=(6, 3 * len(positions)))
    for (correct, i), (original, overlay) in zip(sorted(positions.items(), reverse=True), axes):
        image, label = dataset[i]
        index = dataset.indices[i]
        raw_image, raw_label = raw[index]
        with torch.inference_mode():
            prediction = int(model(image.unsqueeze(0)).argmax(1)[0])
            weights = model.get_attention_weights()[-1]
        patch = config['model']['kwargs']['patch_size']
        height, width = image.shape[-2:]
        grid = (height // patch, width // patch)
        tokens = 1 + grid[0] * grid[1]
        if (weights.ndim != 4 or weights.shape[-2:] != (tokens, tokens)
                or raw_image.size != (width, height) or (raw_label, prediction) !=
                (label, result['predictions'][i]) or (label == prediction) != correct):
            raise ValueError('Attention grid or validation sample does not match the checkpoint')
        heat = weights[0, :, 0, 1:].mean(0).reshape(grid).numpy()
        heat = np.repeat(np.repeat(heat, patch, axis=0), patch, axis=1)
        original.imshow(raw_image)
        original.set_title(f'Index {index} | true: {result["class_names"][label]}')
        overlay.imshow(raw_image)
        overlay.imshow(heat, cmap='magma', alpha=0.48, interpolation='nearest')
        overlay.set_title(f'Prediction: {result["class_names"][prediction]}\nLast block, mean over heads')
        for axis in (original, overlay):
            axis.axis('off')
    save_figure(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', choices=('pilot', 'validation'), default='pilot')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for number, directory in enumerate(args.run, 1):
        config, result, row, history = inspect_run(directory)
        key = tuple(row[field] for field in ('architecture', 'ratio', 'split_seed', 'train_seed'))
        if key in seen:
            raise ValueError(f'Duplicate run for {key}')
        seen.add(key)
        records.append(row)
        name = f'{number}_{row["architecture"]}_{row["ratio"]}'
        fig, axis = plt.subplots()
        for split in ('train', 'val'):
            axis.plot([int(h['epoch']) for h in history],
                      [float(h[f'{split}_accuracy']) for h in history], 'o-', label=split)
        axis.set(xlabel='Epoch', ylabel='Accuracy', title=f'{args.label}: {name}')
        axis.legend()
        save_figure(fig, args.out / f'{name}_curve.png')
        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (result['labels'], result['predictions']), 1)
        fig, axis = plt.subplots(figsize=(8, 6))
        fig.colorbar(axis.imshow(confusion, cmap='Blues'), ax=axis)
        axis.set(xlabel='Predicted', ylabel='True', title=f'{args.label}: {name}')
        axis.set_xticks(range(10), result['class_names'], rotation=45)
        axis.set_yticks(range(10), result['class_names'])
        save_figure(fig, args.out / f'{name}_confusion.png')
        if row['architecture'] == 'vit':
            attention_examples(directory, config, result, args.out / f'{name}_attention.png')
    with (args.out / 'validation.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scope', *records[0]])
        writer.writeheader()
        writer.writerows({'scope': args.label, **row} for row in records)
    fig, axis = plt.subplots()
    for row in records:
        model = row['architecture']
        axis.scatter(float(row['ratio']) * 100, row['accuracy'],
                     marker='o' if model == 'vit' else 's', label=model)
    labels = dict(zip(*reversed(axis.get_legend_handles_labels())))
    axis.legend(labels.values(), labels.keys())
    axis.set(xlabel='Share of the 45,000 training images (%)', ylabel='Validation accuracy',
             title=f'{args.label}: available runs')
    axis.set_xticks([10, 50, 100])
    save_figure(fig, args.out / 'accuracy_data.png')


if __name__ == '__main__':
    main()
