"""Real-data integration pilot; never evaluates the official test set.

Run from the repository root: python -X utf8 -m scripts.check_hien_week1
Outputs are explicitly pilot evidence, not the main experiment table.
"""
import argparse
import json
import shutil
import platform
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from src.data import build_loaders, build_test_loader, load_manifest, make_splits, validate_manifest
from src.evaluate import evaluate_checkpoint, build_run_row, append_run
from src.models.resnet import build_model
from src.train import fit, seed_everything
from scripts.summarize import load_runs, summarize, save_summary_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/hien_week1_smoke')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(Path('configs/hien_smoke.yaml').read_text())
    train, val = build_loaders(config)
    test_samples = len(build_test_loader(config).dataset)
    targets = train.dataset.dataset.targets
    checks = {}
    for seed in (2026, 2027, 2028):
        manifest = load_manifest(seed)
        validate_manifest(manifest, targets, seed)
        regenerated = make_splits(seed)
        for key in ('train_indices', 'val_indices', 'subsets'):
            if regenerated[key] != manifest[key]:
                raise ValueError(f'Seed {seed}: manifest does not reproduce')
        checks[str(seed)] = {'stratified': True, 'reproduced': True,
                            'train': 45000, 'val': 5000, 'subsets': [4500, 22500, 45000]}
    # Rebuild after inspecting batches so pilot starts at the documented generator state.
    images, labels = next(iter(train))
    if images.shape != (128, 3, 32, 32) or labels.dtype != torch.int64:
        raise ValueError('Unexpected batch contract')
    train, val = build_loaders(config)
    seed_everything(config['seed'])
    model = build_model(config)
    run_dir = Path(args.run_dir)
    fit(model, train, val, config, run_dir, device=args.device)
    result = evaluate_checkpoint(run_dir / 'best.pt', config, 'val', args.device)
    state = torch.load(run_dir / 'best.pt', weights_only=True, map_location='cpu')
    if abs(result['accuracy'] - state['best_val_accuracy']) > 1e-12:
        raise ValueError('Evaluator and trainer disagree')
    (run_dir / 'predictions_val.json').write_text(json.dumps(result), encoding='utf-8')
    row = build_run_row(run_dir / 'config.yaml', run_dir / 'best.pt', result, 'val',
                        notes='One-epoch integration pilot; not final experiment results')
    # A run-local CSV avoids accidentally appending a second copy of the same seed.
    append_run(row, run_dir / 'evaluations.csv')
    save_summary_csv(summarize(load_runs(run_dir / 'evaluations.csv', 'val')),
                     run_dir / 'summary.csv')
    for source, target in [('evaluations.csv', 'hien_pilot_runs.csv'),
                           ('summary.csv', 'hien_pilot_summary.csv'),
                           ('metrics.csv', 'hien_pilot_metrics.csv')]:
        shutil.copyfile(run_dir / source, Path('results') / target)
    evidence = {'purpose': 'week-one integration pilot, not final comparison',
                'device': args.device, 'torch': str(torch.__version__),
                'python': platform.python_version(),
                'packages': {name: version(name) for name in ('torchvision', 'numpy', 'PyYAML', 'pytest', 'tensorboard')},
                'test_evaluated': False, 'test_samples': test_samples, 'splits': checks,
                'batch_shape': list(images.shape), 'accuracy': result['accuracy'],
                'correct': result['correct'], 'total': result['total'],
                'trainer_evaluator_match': True, 'run_dir': str(run_dir),
                'manifest_sha256': state['manifest_sha256']}
    Path('results/hien_week1_checks.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
