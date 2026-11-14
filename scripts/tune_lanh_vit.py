"""Thực hiện khảo sát siêu tham số (hyperparameter tuning) cho ViT — Tuần 2 Ngày 8.

Khảo sát 3 công thức huấn luyện trên tập 10% (4.500 ảnh, 3 epochs) với split_seed=2026, seed=42:
- Recipe 1: SGD (lr=0.05, momentum=0.9, wd=0.0005) - theo pilot
- Recipe 2: SGD (lr=0.02, momentum=0.9, wd=0.0005) - tốc độ học vừa phải
- Recipe 3: AdamW (lr=0.001, wd=0.01) - chuẩn Transformer

Kết quả được ghi nhận để nhóm khóa recipe cho các lượt chạy chính Tuần 2 (T09, T10).
"""

import copy
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

from src.data import build_loaders
from src.evaluate import evaluate_checkpoint
from src.models.vit import build_model
from src.train import fit, seed_everything


def run_candidate(name, train_settings, base_config, device='cpu'):
    print(f"\n--- Thử nghiệm cấu hình: {name} ---")
    config = copy.deepcopy(base_config)
    config['training'] = train_settings
    run_dir = Path(f'runs/tuning_{name}')
    if run_dir.exists():
        shutil.rmtree(run_dir)

    train_loader, val_loader = build_loaders(config)
    seed_everything(config['seed'])
    model = build_model(config)

    t0 = time.perf_counter()
    history = fit(model, train_loader, val_loader, config, run_dir, device=device)
    elapsed = time.perf_counter() - t0

    eval_result = evaluate_checkpoint(run_dir / 'best.pt', config, 'val', device)

    best_val_acc = eval_result['accuracy']
    final_train_acc = history[-1]['train_accuracy']
    final_val_loss = history[-1]['val_loss']

    print(f"  Hoàn thành trong {elapsed:.1f}s")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Final Train Accuracy: {final_train_acc:.4f} | Final Val Loss: {final_val_loss:.4f}")

    return {
        'name': name,
        'optimizer': train_settings.get('optimizer', 'sgd'),
        'lr': train_settings['lr'],
        'weight_decay': train_settings.get('weight_decay', 0),
        'best_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'final_val_loss': final_val_loss,
        'elapsed_seconds': round(elapsed, 2),
    }


def main():
    torch.set_num_threads(4)
    base_config = yaml.safe_load(Path('configs/vit.yaml').read_text(encoding='utf-8'))
    base_config['training']['epochs'] = 3

    candidates = [
        ('sgd_lr005', {'epochs': 3, 'optimizer': 'sgd', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 0.0005}),
        ('sgd_lr002', {'epochs': 3, 'optimizer': 'sgd', 'lr': 0.02, 'momentum': 0.9, 'weight_decay': 0.0005}),
        ('adamw_lr001', {'epochs': 3, 'optimizer': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}),
    ]

    results = []
    for name, settings in candidates:
        res = run_candidate(name, settings, base_config, device='cpu')
        results.append(res)

    print("\n" + "=" * 65)
    print("KẾT QUẢ KHẢO SÁT SIÊU THAM SỐ ViT (10% data, 3 epochs)")
    print("=" * 65)
    print(f"{'Cấu hình':<15} {'Optimizer':<10} {'LR':<8} {'Best Val Acc':<14} {'Train Acc':<12} {'Thời gian'}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<15} {r['optimizer']:<10} {r['lr']:<8} {r['best_val_acc']*100:>6.2f}%       {r['final_train_acc']*100:>6.2f}%      {r['elapsed_seconds']}s")

    best = max(results, key=lambda x: x['best_val_acc'])
    print(f"\n=> Cấu hình tốt nhất: {best['name']} với Val Accuracy = {best['best_val_acc']*100:.2f}%")

    out_file = Path('results/lanh_tuning_results.json')
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Đã lưu kết quả tại: {out_file}")


if __name__ == '__main__':
    main()
