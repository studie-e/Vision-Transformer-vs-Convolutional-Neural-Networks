"""Thực hiện các lượt chạy chính Tuần 2 cho Lan Anh — ViT 10%, 50%, 100% dữ liệu (T10).

Seed đầu tiên: split_seed=2026, train_seed=42.
Recipe đã khóa qua tuning:
- Model: ViT nhỏ tự cài (274.282 tham số)
- Optimizer: AdamW, lr=0.001, weight_decay=0.01, CosineAnnealingLR (5 epochs)
- Batch size: 128
- Device: CPU (torch 4 threads)

Đầu ra:
- Thư mục runs: runs/vit_d10_s42, runs/vit_d50_s42, runs/vit_d100_s42
- Kết quả lưu tại: results/runs.csv và results/lanh_week2_runs.csv
- Tổng hợp tại: results/lanh_week2_summary.csv
- Mẫu attention trích xuất cho Hiền tại: results/lanh_week2_attentions.pt
"""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

from scripts.summarize import load_runs, save_summary_csv, summarize
from src.data import build_loaders
from src.evaluate import append_run, build_run_row, evaluate_checkpoint
from src.models.vit import build_model, count_parameters
from src.train import fit, seed_everything


RUN_CONFIGS = [
    {
        'name': 'vit_d10_s42',
        'config_path': 'configs/vit_d10.yaml',
        'ratio': '0.10',
        'desc': '10% dữ liệu (4.500 ảnh train)',
    },
    {
        'name': 'vit_d50_s42',
        'config_path': 'configs/vit_d50.yaml',
        'ratio': '0.50',
        'desc': '50% dữ liệu (22.500 ảnh train)',
    },
    {
        'name': 'vit_d100_s42',
        'config_path': 'configs/vit_d100.yaml',
        'ratio': '1.00',
        'desc': '100% dữ liệu (45.000 ảnh train)',
    },
]


def extract_sample_attentions(model, val_loader, device):
    """Trích xuất attention weights trên 10 ảnh cố định đại diện cho 10 lớp CIFAR-10."""
    model.eval()
    samples_per_class = {}
    with torch.no_grad():
        for images, labels in val_loader:
            for img, lbl in zip(images, labels):
                cls_idx = lbl.item()
                if cls_idx not in samples_per_class:
                    samples_per_class[cls_idx] = img
                if len(samples_per_class) == 10:
                    break
            if len(samples_per_class) == 10:
                break

    sorted_classes = sorted(samples_per_class.keys())
    sample_tensor = torch.stack([samples_per_class[c] for c in sorted_classes]).to(device)
    with torch.no_grad():
        logits = model(sample_tensor)
    weights = model.get_attention_weights()

    return {
        'images': sample_tensor.cpu(),
        'labels': torch.tensor(sorted_classes),
        'logits': logits.cpu(),
        'attention_layers': [w.cpu() for w in weights],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    torch.set_num_threads(4)
    print("=" * 70)
    print("TIẾN HÀNH 3 LƯỢT CHẠY CHÍNH TUẦN 2 — LAN ANH (T10)")
    print(f"Thiết bị: {args.device} | Mô hình: ViT tự cài (274.282 tham số)")
    print(f"Seed đầu: split_seed=2026, train_seed=42 | Epochs: 5 | Optimizer: AdamW lr=0.001")
    print("=" * 70)

    results_overview = []
    attention_bundle = {}

    runs_csv_path = Path('results/runs.csv')
    week2_csv_path = Path('results/lanh_week2_runs.csv')

    for run_spec in RUN_CONFIGS:
        name = run_spec['name']
        cfg_file = run_spec['config_path']
        desc = run_spec['desc']
        run_dir = Path('runs') / name

        print(f"\n=======================================================")
        print(f"Bắt đầu lượt chạy: {name} ({desc})")
        print(f"Config: {cfg_file} -> Thư mục run: {run_dir}")
        print(f"=======================================================")

        if run_dir.exists():
            shutil.rmtree(run_dir)

        config = yaml.safe_load(Path(cfg_file).read_text(encoding='utf-8'))
        train_loader, val_loader = build_loaders(config)
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"  DataLoader nạp: {n_train:,} ảnh train, {n_val:,} ảnh validation")

        seed_everything(config['seed'])
        model = build_model(config)
        params = count_parameters(model)

        t_start = time.perf_counter()
        history = fit(model, train_loader, val_loader, config, run_dir, device=args.device)
        total_time = time.perf_counter() - t_start

        best_pt = run_dir / 'best.pt'
        eval_result = evaluate_checkpoint(best_pt, config, 'val', args.device)
        state = torch.load(best_pt, weights_only=True, map_location='cpu')

        diff = abs(eval_result['accuracy'] - state['best_val_accuracy'])
        if diff > 1e-7:
            raise ValueError(f"Sai lệch giữa evaluator và trainer: {diff}")

        # Save predictions
        (run_dir / 'predictions_val.json').write_text(
            json.dumps(eval_result, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        # Build row
        row = build_run_row(
            run_dir / 'config.yaml',
            best_pt,
            eval_result,
            'val',
            notes=f'ViT Week-2 main run {name} ({desc})'
        )

        append_run(row, run_dir / 'evaluations.csv')
        append_run(row, week2_csv_path)
        append_run(row, runs_csv_path)

        # Extract attention weights on 10 fixed class samples
        attn_data = extract_sample_attentions(model, val_loader, args.device)
        attention_bundle[name] = attn_data

        best_val_acc = eval_result['accuracy']
        final_train_acc = history[-1]['train_accuracy']
        print(f"  Hoàn thành trong: {total_time:.1f}s ({total_time/60:.2f} phút)")
        print(f"  Best Val Accuracy: {best_val_acc*100:.2f}% (Epoch {state['best_epoch']})")
        print(f"  Final Train Accuracy: {final_train_acc*100:.2f}%")

        results_overview.append({
            'run': name,
            'ratio': run_spec['ratio'],
            'train_samples': n_train,
            'best_val_acc': best_val_acc,
            'best_epoch': state['best_epoch'],
            'final_train_acc': final_train_acc,
            'total_seconds': round(total_time, 2),
            'history': history,
        })

    # Save all sample attention weights for Hien
    attn_save_path = Path('results/lanh_week2_attentions.pt')
    torch.save(attention_bundle, attn_save_path)
    print(f"\nĐã lưu ma trận attention mẫu cho Hiền tại: {attn_save_path}")

    # Generate summary
    loaded = load_runs(week2_csv_path)
    summary_rows = summarize(loaded)
    summary_path = Path('results/lanh_week2_summary.csv')
    save_summary_csv(summary_rows, summary_path)
    print(f"Đã lưu bảng tổng hợp tại: {summary_path}")

    # Save detailed overview json
    overview_path = Path('results/lanh_week2_overview.json')
    overview_path.write_text(json.dumps(results_overview, ensure_ascii=False, indent=2), encoding='utf-8')

    print("\n" + "=" * 70)
    print("TỔNG KẾT KẾT QUẢ 3 LƯỢT CHẠY ViT TUẦN 2 (T10)")
    print("=" * 70)
    print(f"{'Lượt chạy':<15} {'Tỷ lệ':<8} {'Ảnh train':<12} {'Best Val Acc':<15} {'Train Acc':<12} {'Thời gian'}")
    print("-" * 70)
    for res in results_overview:
        print(f"{res['run']:<15} {res['ratio']:<8} {res['train_samples']:<12,} {res['best_val_acc']*100:>6.2f}% (ep {res['best_epoch']})   {res['final_train_acc']*100:>6.2f}%      {res['total_seconds']/60:.1f}m")
    print("=" * 70)


if __name__ == '__main__':
    main()
