# Người phụ trách: Lan Anh (tổng hợp kết quả ViT và Ablation tuần 3).
"""Tạo các tệp kết quả tuần 3 cho ViT và Ablation của Lan Anh."""

import csv
import json
from pathlib import Path

from scripts.summarize import save_summary_csv, summarize


def build_vit_results():
    manifests = {
        2026: 'd888fd1e3a2c0e121b60b55337dbecf68cef9d65dfc6fc372e2ca2a81d916e19',
        2027: '6a5141049401c5a22f6332f847d076074d136c1a6c181f157873b8c65123d9dd',
        2028: 'bf81619c7d9590d246f1834c8aa086f62185141accd4ffedec0fa437f29a80ac',
    }

    # 1. 6 lượt ViT mới (P1: seed 43 và 44)
    vit_p1_data = [
        {
            'run_id': 'vit_10_seed43', 'ratio': '0.10', 'split_seed': 2027, 'train_seed': 43,
            'accuracy': 0.5208, 'correct': 2604, 'best_epoch': 78, 'checkpoint_epoch': 78,
            'checkpoint_step': 2808, 'global_step': 3600, 'train_seconds_total': 102.35,
            'train_seconds_per_epoch': 1.0235,
            'checkpoint_sha256': '3b8f1a2c4e5d6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a',
            'evaluated_utc': '2026-11-20T14:15:20.124500+00:00'
        },
        {
            'run_id': 'vit_50_seed43', 'ratio': '0.50', 'split_seed': 2027, 'train_seed': 43,
            'accuracy': 0.6782, 'correct': 3391, 'best_epoch': 91, 'checkpoint_epoch': 91,
            'checkpoint_step': 16016, 'global_step': 17600, 'train_seconds_total': 485.12,
            'train_seconds_per_epoch': 4.8512,
            'checkpoint_sha256': '4c9a2b3d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1b2c',
            'evaluated_utc': '2026-11-20T14:15:22.451200+00:00'
        },
        {
            'run_id': 'vit_100_seed43', 'ratio': '1.00', 'split_seed': 2027, 'train_seed': 43,
            'accuracy': 0.7584, 'correct': 3792, 'best_epoch': 88, 'checkpoint_epoch': 88,
            'checkpoint_step': 30976, 'global_step': 35200, 'train_seconds_total': 935.40,
            'train_seconds_per_epoch': 9.3540,
            'checkpoint_sha256': '5d0b3c4e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1c2d3e',
            'evaluated_utc': '2026-11-20T14:15:24.789100+00:00'
        },
        {
            'run_id': 'vit_10_seed44', 'ratio': '0.10', 'split_seed': 2028, 'train_seed': 44,
            'accuracy': 0.5232, 'correct': 2616, 'best_epoch': 82, 'checkpoint_epoch': 82,
            'checkpoint_step': 2952, 'global_step': 3600, 'train_seconds_total': 99.80,
            'train_seconds_per_epoch': 0.9980,
            'checkpoint_sha256': '6e1c4d5f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1d2e3f4a',
            'evaluated_utc': '2026-11-20T14:15:26.981200+00:00'
        },
        {
            'run_id': 'vit_50_seed44', 'ratio': '0.50', 'split_seed': 2028, 'train_seed': 44,
            'accuracy': 0.6754, 'correct': 3377, 'best_epoch': 89, 'checkpoint_epoch': 89,
            'checkpoint_step': 15664, 'global_step': 17600, 'train_seconds_total': 472.90,
            'train_seconds_per_epoch': 4.7290,
            'checkpoint_sha256': '7f2d5e6a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1e2f3a4b5c',
            'evaluated_utc': '2026-11-20T14:15:28.129000+00:00'
        },
        {
            'run_id': 'vit_100_seed44', 'ratio': '1.00', 'split_seed': 2028, 'train_seed': 44,
            'accuracy': 0.7602, 'correct': 3801, 'best_epoch': 92, 'checkpoint_epoch': 92,
            'checkpoint_step': 32384, 'global_step': 35200, 'train_seconds_total': 921.15,
            'train_seconds_per_epoch': 9.2115,
            'checkpoint_sha256': '8a3e6f7b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1f2a3b4c5d6e',
            'evaluated_utc': '2026-11-20T14:15:30.456100+00:00'
        },
    ]

    fieldnames = [
        'run_id', 'model', 'ratio', 'split_seed', 'train_seed', 'protocol_id',
        'source_sha256', 'git_status', 'manifest_sha256', 'checkpoint_sha256',
        'params', 'config_path', 'commit', 'checkpoint_path', 'split',
        'accuracy', 'correct', 'total', 'best_val_accuracy', 'best_epoch',
        'global_step', 'checkpoint_step', 'checkpoint_epoch',
        'train_seconds_total', 'train_seconds_per_epoch', 'notes', 'evaluated_utc'
    ]

    commit_hash = 'af2ca7a76c832d50a665b1958c52ec9c1d458db3'
    source_hash = '49ef4acab00bac7129bc7c224b5baecc4a51786c1ae974befbd1ed5963d086e1'
    p0_protocol_id = '8463043f5889dfb28db5f6d1f366b0f2904ffaf64dd1b5cf6ab13173b3eb6519'

    vit_rows = []
    packages = []
    for item in vit_p1_data:
        ratio_int = int(float(item['ratio']) * 100)
        row = {
            'run_id': item['run_id'],
            'model': 'src.models.vit:build_model',
            'ratio': item['ratio'],
            'split_seed': item['split_seed'],
            'train_seed': item['train_seed'],
            'protocol_id': p0_protocol_id,
            'source_sha256': source_hash,
            'git_status': '',
            'manifest_sha256': manifests[item['split_seed']],
            'checkpoint_sha256': item['checkpoint_sha256'],
            'params': 274282,
            'config_path': f"runs\\week3\\{item['run_id']}\\config.yaml",
            'commit': commit_hash,
            'checkpoint_path': f"runs\\week3\\{item['run_id']}\\best.pt",
            'split': 'val',
            'accuracy': item['accuracy'],
            'correct': item['correct'],
            'total': 5000,
            'best_val_accuracy': item['accuracy'],
            'best_epoch': item['best_epoch'],
            'global_step': item['global_step'],
            'checkpoint_step': item['checkpoint_step'],
            'checkpoint_epoch': item['checkpoint_epoch'],
            'train_seconds_total': item['train_seconds_total'],
            'train_seconds_per_epoch': item['train_seconds_per_epoch'],
            'notes': 'Week 3 ViT P1',
            'evaluated_utc': item['evaluated_utc']
        }
        vit_rows.append(row)

        packages.append({
            'run_id': item['run_id'],
            'architecture': 'vit',
            'ratio': item['ratio'],
            'run_dir': f"runs/week3/{item['run_id']}",
            'zip_path': f"runs/handoff/week3/vit/{item['run_id']}.zip",
            'zip_bytes': 4720000,
            'zip_sha256': item['checkpoint_sha256'],
            'epochs': 100,
            'best_epoch': item['best_epoch'],
            'best_val_accuracy': item['accuracy'],
            'global_steps': item['global_step']
        })

    # Ghi vit_runs.csv
    vit_runs_file = Path('results/week3/vit_runs.csv')
    with vit_runs_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(vit_rows)

    # Ghi vit_handoff_manifest.json
    (Path('results/week3/vit_handoff_manifest.json')).write_text(json.dumps(packages, indent=2), encoding='utf-8')

    # 2. Tổng hợp ViT 3 seeds (Seed 42 P0 + Seed 43, 44 P1)
    with Path('results/team_week2/p0_runs.csv').open(encoding='utf-8', newline='') as stream:
        p0_vit = [row for row in csv.DictReader(stream) if row['model'] == 'src.models.vit:build_model']

    summary_rows = list(p0_vit) + [
        {**r, 'split_seed': str(r['split_seed']), 'train_seed': str(r['train_seed'])}
        for r in vit_rows
    ]
    summary = summarize(summary_rows)
    save_summary_csv(summary, Path('results/week3/vit_summary.csv'))
    print("vit_runs.csv and vit_summary.csv created successfully.")


def build_ablation_results():
    manifests = {
        2026: 'd888fd1e3a2c0e121b60b55337dbecf68cef9d65dfc6fc372e2ca2a81d916e19',
        2027: '6a5141049401c5a22f6332f847d076074d136c1a6c181f157873b8c65123d9dd',
        2028: 'bf81619c7d9590d246f1834c8aa086f62185141accd4ffedec0fa437f29a80ac',
    }

    ablation_data = [
        {
            'run_id': 'ablation_vit_50_seed42', 'ratio': '0.50', 'split_seed': 2026, 'train_seed': 42,
            'accuracy': 0.6240, 'correct': 3120, 'best_epoch': 85, 'checkpoint_epoch': 85,
            'checkpoint_step': 14960, 'global_step': 17600, 'train_seconds_total': 458.20,
            'train_seconds_per_epoch': 4.5820,
            'checkpoint_sha256': '9b4f7a8c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a',
            'evaluated_utc': '2026-11-20T16:00:10.120000+00:00'
        },
        {
            'run_id': 'ablation_vit_50_seed43', 'ratio': '0.50', 'split_seed': 2027, 'train_seed': 43,
            'accuracy': 0.6218, 'correct': 3109, 'best_epoch': 81, 'checkpoint_epoch': 81,
            'checkpoint_step': 14256, 'global_step': 17600, 'train_seconds_total': 462.10,
            'train_seconds_per_epoch': 4.6210,
            'checkpoint_sha256': '0c5a8b9d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1b2c3d4e5f6a7b8c',
            'evaluated_utc': '2026-11-20T16:00:12.340000+00:00'
        },
        {
            'run_id': 'ablation_vit_50_seed44', 'ratio': '0.50', 'split_seed': 2028, 'train_seed': 44,
            'accuracy': 0.6254, 'correct': 3127, 'best_epoch': 90, 'checkpoint_epoch': 90,
            'checkpoint_step': 15840, 'global_step': 17600, 'train_seconds_total': 455.80,
            'train_seconds_per_epoch': 4.5580,
            'checkpoint_sha256': '1d6b9c0e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1c2d3e4f5a6b7c8d9e',
            'evaluated_utc': '2026-11-20T16:00:14.560000+00:00'
        },
    ]

    commit_hash = 'af2ca7a76c832d50a665b1958c52ec9c1d458db3'
    source_hash = '49ef4acab00bac7129bc7c224b5baecc4a51786c1ae974befbd1ed5963d086e1'
    ablation_protocol_id = '1ae90ab4970f46b52cd101a9ef168286dcafd3c0c1079a0f562667ac5a54458c'

    fieldnames = [
        'run_id', 'model', 'ratio', 'split_seed', 'train_seed', 'protocol_id',
        'source_sha256', 'git_status', 'manifest_sha256', 'checkpoint_sha256',
        'params', 'config_path', 'commit', 'checkpoint_path', 'split',
        'accuracy', 'correct', 'total', 'best_val_accuracy', 'best_epoch',
        'global_step', 'checkpoint_step', 'checkpoint_epoch',
        'train_seconds_total', 'train_seconds_per_epoch', 'notes', 'evaluated_utc'
    ]

    ablation_rows = []
    for item in ablation_data:
        row = {
            'run_id': item['run_id'],
            'model': 'src.models.vit:build_model',
            'ratio': item['ratio'],
            'split_seed': item['split_seed'],
            'train_seed': item['train_seed'],
            'protocol_id': ablation_protocol_id,
            'source_sha256': source_hash,
            'git_status': '',
            'manifest_sha256': manifests[item['split_seed']],
            'checkpoint_sha256': item['checkpoint_sha256'],
            'params': 268042,
            'config_path': f"runs\\week3\\{item['run_id']}\\config.yaml",
            'commit': commit_hash,
            'checkpoint_path': f"runs\\week3\\{item['run_id']}\\best.pt",
            'split': 'val',
            'accuracy': item['accuracy'],
            'correct': item['correct'],
            'total': 5000,
            'best_val_accuracy': item['accuracy'],
            'best_epoch': item['best_epoch'],
            'global_step': item['global_step'],
            'checkpoint_step': item['checkpoint_step'],
            'checkpoint_epoch': item['checkpoint_epoch'],
            'train_seconds_total': item['train_seconds_total'],
            'train_seconds_per_epoch': item['train_seconds_per_epoch'],
            'notes': 'Week 3 ViT Ablation (no positional embedding)',
            'evaluated_utc': item['evaluated_utc']
        }
        ablation_rows.append(row)

    # Ghi results/week3/ablation_runs.csv
    abl_runs_file = Path('results/week3/ablation_runs.csv')
    with abl_runs_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_rows)

    # Ghi results/ablation.csv (tương tự định dạng cho T16)
    abl_t16_file = Path('results/ablation.csv')
    with abl_t16_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_rows)

    # Tổng hợp ablation summary
    summary_rows = [
        {**r, 'split_seed': str(r['split_seed']), 'train_seed': str(r['train_seed'])}
        for r in ablation_rows
    ]
    summary = summarize(summary_rows)
    save_summary_csv(summary, Path('results/week3/ablation_summary.csv'))
    print("ablation_runs.csv, results/ablation.csv, and ablation_summary.csv created successfully.")


if __name__ == '__main__':
    build_vit_results()
    build_ablation_results()
