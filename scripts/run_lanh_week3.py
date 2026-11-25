# Lan Anh phụ trách mã và kết quả ViT/ablation. Minh hỗ trợ vận hành GPU.
# Người phụ trách: Lan Anh (các lượt ViT và Ablation tuần 3).
"""Chạy và kiểm tra các lượt ViT và Ablation của Lan Anh trong tuần 3."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from scripts.run_minh_week3 import verify_training_source
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.package_team_week2 import package_run
from scripts.run_minh_week2 import completed_state, run_one
from scripts.summarize import save_summary_csv, summarize
from src.evaluate import append_run, build_run_row, evaluate_checkpoint, protocol_id


CONFIG_DIR = Path('configs/week3')
RUN_DIR = Path('runs/week3')
RESULT_DIR = Path('results/week3')


# Hàm config_paths trả danh sách YAML theo thứ tự seed rồi đến tỷ lệ.
def config_paths(model):
    if model == 'vit':
        return [CONFIG_DIR / f'vit_{ratio}_seed{seed}.yaml'
                for seed in (43, 44) for ratio in (10, 50, 100)]
    return [CONFIG_DIR / f'ablation_vit_50_seed{seed}.yaml' for seed in (42, 43, 44)]


# Hàm collect_validation nạp lại checkpoint, kiểm tra tính hợp lệ và tổng hợp bảng kết quả.
def collect_validation(model, device='cuda'):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = config_paths(model)
    destination = RESULT_DIR / f'{model}_runs.csv'
    temporary = destination.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    package_dir = Path('runs/handoff/week3') / model
    rows = []
    packages = []

    # Nếu run_dir tồn tại trên máy (ví dụ sau khi train trên GPU), thực hiện nạp và đối chiếu
    all_runs_present = all((RUN_DIR / p.stem / 'best.pt').is_file() for p in paths)
    if all_runs_present:
        states = [completed_state(path, RUN_DIR / path.stem) for path in paths]
        for path, state in zip(paths, states, strict=True):
            folder = RUN_DIR / path.stem
            config = state['config']
            result = evaluate_checkpoint(folder / 'best.pt', config, split='val', device=device)
            if result['total'] != 5000 or result['accuracy'] != state['best_val_accuracy']:
                raise ValueError(f'Validation does not match training log: {folder}')
            events = EventAccumulator(str(folder / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
            if [item.step for item in events.Scalars('val_accuracy')] != list(range(1, 101)):
                raise ValueError(f'TensorBoard does not contain epochs 1–100: {folder}')
            (folder / 'predictions_val.json').write_text(json.dumps(result), encoding='utf-8')
            note = 'Week 3 ViT P1' if model == 'vit' else 'Week 3 ViT Ablation (no positional embedding)'
            row = build_run_row(folder / 'config.yaml', folder / 'best.pt', result, 'val', note)
            rows.append(row)
            append_run(row, temporary)
            (RESULT_DIR / f'{folder.name}_metrics.csv').write_bytes((folder / 'metrics.csv').read_bytes())
            packages.append(package_run(folder, package_dir, expected_config_path=path))
            print(f"[VAL] {folder.name}: {result['correct']}/{result['total']}", flush=True)

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

        summary = summarize(summary_rows)
        expected_groups = 3 if model == 'vit' else 1
        if len(summary) != expected_groups or any(row['n_seeds'] != 3 for row in summary):
            raise ValueError('Thiếu seed hoặc có cấu hình không khớp trong bảng validation')
        save_summary_csv(summary, RESULT_DIR / f'{model}_summary.csv')
        temporary.replace(destination)
        if model == 'ablation':
            (RESULT_DIR.parent / 'ablation.csv').write_bytes(destination.read_bytes())
        manifest = json.dumps(packages, indent=2)
        (package_dir / 'handoff_manifest.json').write_text(manifest, encoding='utf-8')
        (RESULT_DIR / f'{model}_handoff_manifest.json').write_text(manifest, encoding='utf-8')
    else:
        missing = [str(RUN_DIR / p.stem / 'best.pt') for p in paths
                   if not (RUN_DIR / p.stem / 'best.pt').is_file()]
        raise FileNotFoundError(
            'Collect requires original checkpoints for independent validation. Missing: '
            + ', '.join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'collect'), required=True)
    parser.add_argument('--model', choices=('vit', 'ablation'), default='vit')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.stage == 'train':
        for path in config_paths(args.model):
            run_one(path, RUN_DIR / path.stem, args.device, resume=True)
    else:
        collect_validation(args.model, args.device)


if __name__ == '__main__':
    main()
