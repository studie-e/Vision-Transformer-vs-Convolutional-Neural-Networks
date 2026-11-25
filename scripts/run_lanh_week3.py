# Người phụ trách: Lan Anh (các lượt ViT và Ablation tuần 3–4).
"""Chạy, kiểm tra độc lập bằng checkpoint thật và đóng gói các lượt ViT/Ablation của Lan Anh."""

import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.package_team_week2 import package_run
from scripts.run_minh_week2 import completed_state, run_one
from scripts.summarize import save_summary_csv, summarize
from src.evaluate import append_run, build_run_row, evaluate_checkpoint, protocol_id


CONFIG_DIR = Path('configs/week3')
RUN_DIR = Path('runs/week3')
RESULT_DIR = Path('results/week3')
NON_TRAINING_SOURCES = {'src/visualize.py', 'src/predict.py'}


def config_paths(model, ratios=None, seeds=None):
    """Trả danh sách YAML theo thứ tự seed rồi đến tỷ lệ (có thể lọc theo ratios/seeds)."""
    if model == 'vit':
        use_seeds = tuple(seeds) if seeds else (43, 44)
        use_ratios = tuple(ratios) if ratios else (10, 50, 100)
        return [CONFIG_DIR / f'vit_{ratio}_seed{seed}.yaml'
                for seed in use_seeds for ratio in use_ratios]
    use_seeds = tuple(seeds) if seeds else (42, 43, 44)
    return [CONFIG_DIR / f'ablation_vit_50_seed{seed}.yaml' for seed in use_seeds]


def verify_training_source(baseline, repeat):
    """Kiểm tra tính nhất quán AST của mã nguồn huấn luyện giữa P0 và P1."""
    for row in (baseline, repeat):
        if any(line[3:].startswith(('src/', 'requirements.txt'))
               for line in row['git_status'].splitlines()):
            raise ValueError(f"Uncommitted training source for {row['run_id']}")
    changed = subprocess.check_output(
        ['git', 'diff', '--no-renames', '--name-only', baseline['commit'], repeat['commit'],
         '--', 'src', 'requirements.txt'],
        text=True,
    ).splitlines()
    for path in changed:
        if path in NON_TRAINING_SOURCES:
            continue
        try:
            before, after = (subprocess.check_output(['git', 'show', f"{row['commit']}:{path}"],
                                                     stderr=subprocess.DEVNULL)
                             for row in (baseline, repeat))
        except subprocess.CalledProcessError as error:
            raise ValueError(f'Different training sources: {path}') from error
        if path.endswith('.py'):
            if ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)):
                continue
        elif path == 'requirements.txt':
            prefixes = ('# Người phụ trách: Minh'.encode('utf-8'),
                        '# Người phụ trách: Lan Anh'.encode('utf-8'))
            before_lines, after_lines = before.splitlines(), after.splitlines()
            if before_lines and before_lines[0].startswith(prefixes):
                before_lines = before_lines[1:]
            if after_lines and after_lines[0].startswith(prefixes):
                after_lines = after_lines[1:]
            if before_lines == after_lines:
                continue
        raise ValueError(f'Different training sources: {path}')


def collect_validation(model, device='cuda', ratios=None, seeds=None):
    """Nạp lại best.pt thật, đối chiếu log 100 epoch + TensorBoard và đóng gói ZIP + SHA-256."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = config_paths(model, ratios=ratios, seeds=seeds)
    missing = [str(RUN_DIR / p.stem / 'best.pt') for p in paths
               if not (RUN_DIR / p.stem / 'best.pt').is_file()]
    if missing:
        raise FileNotFoundError(
            'Collect requires original checkpoints for independent validation. Missing: '
            + ', '.join(missing))

    destination = RESULT_DIR / f'{model}_runs.csv'
    temporary = destination.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    package_dir = Path('runs/handoff/week3') / model
    rows = []
    packages = []

    states = [completed_state(path, RUN_DIR / path.stem) for path in paths]
    for path, state in zip(paths, states, strict=True):
        folder = RUN_DIR / path.stem
        config = state['config']
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        result = evaluate_checkpoint(folder / 'best.pt', config, split='val', device=device)
        if result['total'] != 5000 or result['accuracy'] != state['best_val_accuracy']:
            raise ValueError(f'Validation does not match training log: {folder}')
        events = EventAccumulator(str(folder / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
        if [item.step for item in events.Scalars('val_accuracy')] != list(range(1, 101)):
            raise ValueError(f'TensorBoard does not contain epochs 1–100: {folder}')
        (folder / 'predictions_val.json').write_text(json.dumps(result), encoding='utf-8')
        gpu_note = metadata.get('gpu') or metadata.get('device', device)
        torch_note = metadata.get('torch', '')
        base_note = 'Week 3-4 ViT P1' if model == 'vit' else 'Week 3-4 ViT Ablation (no positional embedding)'
        note = f'{base_note} ({gpu_note}, torch {torch_note})'
        row = build_run_row(folder / 'config.yaml', folder / 'best.pt', result, 'val', note)
        rows.append(row)
        append_run(row, temporary)
        (RESULT_DIR / f'{folder.name}_metrics.csv').write_bytes((folder / 'metrics.csv').read_bytes())
        packages.append(package_run(folder, package_dir, expected_config_path=path))
        print(f"[VAL] {folder.name}: {result['correct']}/{result['total']} ({result['accuracy']:.4%})", flush=True)

    summary_rows = rows
    if model == 'vit':
        with Path('results/team_week2/p0_runs.csv').open(encoding='utf-8', newline='') as stream:
            baselines = {row['ratio']: row for row in csv.DictReader(stream)
                         if row['model'] == 'src.models.vit:build_model'}
        summary_rows = list(baselines.values())
        for row, state in zip(rows, states, strict=True):
            baseline = baselines[row['ratio']]
            base_path = Path(f"configs/week2/vit_{int(float(row['ratio']) * 100)}.yaml")
            base_config = yaml.safe_load(base_path.read_text(encoding='utf-8'))
            if protocol_id(base_config) != protocol_id(state['config']):
                raise ValueError(f"Recipe changed since P0: {row['run_id']}")
            verify_training_source(baseline, row)
            summary_rows.append({
                **row,
                'protocol_id': baseline['protocol_id'],
                'split_seed': str(row['split_seed']),
                'train_seed': str(row['train_seed']),
            })
    else:
        # Đồng bộ results/ablation.csv ở thư mục gốc theo đặc tả T16
        Path('results/ablation.csv').write_bytes(temporary.read_bytes())

    summary = summarize(summary_rows)
    save_summary_csv(summary, RESULT_DIR / f'{model}_summary.csv')
    temporary.replace(destination)
    manifest = json.dumps(packages, indent=2)
    (package_dir / 'handoff_manifest.json').write_text(manifest, encoding='utf-8')
    (RESULT_DIR / f'{model}_handoff_manifest.json').write_text(manifest, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'collect'), required=True)
    parser.add_argument('--model', choices=('vit', 'ablation'), default='vit')
    parser.add_argument('--ratios', type=int, nargs='+', choices=(10, 50, 100),
                        help='Tùy chọn lọc tỷ lệ dữ liệu cho ViT (mặc định: cả 10, 50, 100)')
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42, 43, 44),
                        help='Tùy chọn lọc train seed (mặc định: 43, 44 cho ViT; 42, 43, 44 cho Ablation)')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.stage == 'train':
        for path in config_paths(args.model, ratios=args.ratios, seeds=args.seeds):
            run_one(path, RUN_DIR / path.stem, args.device, resume=True)
    else:
        collect_validation(args.model, args.device, ratios=args.ratios, seeds=args.seeds)


if __name__ == '__main__':
    main()
