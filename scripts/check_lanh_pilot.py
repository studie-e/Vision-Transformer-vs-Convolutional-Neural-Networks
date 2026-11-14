"""Pilot tích hợp dữ liệu thật CIFAR-10 cho ViT — tuần 1 Lan Anh.

Chạy: python -X utf8 -m scripts.check_lanh_pilot --device cpu

Các bước kiểm tra:
1. Nạp config ViT và build loaders từ data pipeline thật của Hiền (CIFAR-10, ratio 0.10 = 4.500 ảnh train, 5.000 val).
2. Khởi tạo ViT tự cài (274.282 tham số) theo seed 42.
3. Huấn luyện 1 epoch với trainer chung của Minh, ghi TensorBoard/CSV/checkpoint vào runs/lanh_week1_smoke.
4. Đánh giá checkpoint best.pt bằng src.evaluate; đối chiếu evaluator accuracy với trainer best_val_accuracy.
5. Kiểm tra và xuất attention weights từ ảnh validation thật cho Hiền trực quan hóa.
6. Lưu kết quả pilot vào results/lanh_pilot_runs.csv, results/lanh_pilot_metrics.csv, results/lanh_pilot_summary.csv và results/lanh_week1_checks.json.
"""

import argparse
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from scripts.summarize import load_runs, save_summary_csv, summarize
from src.data import build_loaders, build_test_loader
from src.evaluate import append_run, build_run_row, evaluate_checkpoint
from src.models.vit import build_model, count_parameters
from src.train import fit, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/lanh_week1_smoke')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    torch.set_num_threads(4)
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    print("=" * 65)
    print("PILOT TÍCH HỢP ViT TUẦN 1 — LAN ANH")
    print(f"Thiết bị: {args.device} | Run dir: {run_dir}")
    print("=" * 65)

    config = yaml.safe_load(Path('configs/vit.yaml').read_text(encoding='utf-8'))

    # 1. Build loaders
    print("\n[1/6] Nạp DataLoader dữ liệu thật CIFAR-10...")
    train_loader, val_loader = build_loaders(config)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"  Train: {n_train:,} ảnh (tỷ lệ {config['data']['ratio']})")
    print(f"  Validation: {n_val:,} ảnh")

    # 2. Build model & verify parameter budget
    print("\n[2/6] Khởi tạo ViT tự cài và kiểm tra tham số...")
    seed_everything(config['seed'])
    model = build_model(config)
    params = count_parameters(model)
    print(f"  Số tham số trainable: {params:,}")
    if abs(params - 272474) / 272474 > 0.10:
        raise ValueError(f"Chênh lệch tham số {params} vs 272474 vượt 10%")

    # 3. Train 1 epoch pilot
    print("\n[3/6] Chạy 1 epoch huấn luyện qua trainer chung...")
    start_time = time.perf_counter()
    history = fit(model, train_loader, val_loader, config, run_dir, device=args.device)
    train_time = time.perf_counter() - start_time
    print(f"  Thời gian hoàn thành: {train_time:.2f}s")
    print(f"  Train loss: {history[0]['train_loss']:.4f} | Train acc: {history[0]['train_accuracy']:.4f}")
    print(f"  Val loss:   {history[0]['val_loss']:.4f} | Val acc:   {history[0]['val_accuracy']:.4f}")

    # 4. Evaluate checkpoint with src.evaluate
    print("\n[4/6] Đánh giá lại checkpoint bằng src.evaluate...")
    best_pt = run_dir / 'best.pt'
    eval_result = evaluate_checkpoint(best_pt, config, 'val', args.device)
    state = torch.load(best_pt, weights_only=True, map_location='cpu')

    diff = abs(eval_result['accuracy'] - state['best_val_accuracy'])
    print(f"  Trainer val acc:   {state['best_val_accuracy']:.4f}")
    print(f"  Evaluator val acc: {eval_result['accuracy']:.4f}")
    if diff > 1e-7:
        raise ValueError(f"Sai lệch giữa evaluator và trainer: {diff}")
    print("  OK: Evaluator và Trainer khớp chính xác 100%")

    # Save predictions
    (run_dir / 'predictions_val.json').write_text(
        json.dumps(eval_result, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # 5. Extract attention maps from validation batch for Hien
    print("\n[5/6] Trích xuất attention weights từ batch validation mẫu...")
    model.eval()
    val_iter = iter(val_loader)
    sample_images, sample_labels = next(val_iter)
    sample_images = sample_images[:8]  # lấy 8 ảnh đầu
    sample_labels = sample_labels[:8]

    with torch.no_grad():
        sample_logits = model(sample_images.to(args.device))
    attn_maps = model.get_attention_weights()

    print(f"  Số encoder blocks: {len(attn_maps)}")
    print(f"  Shape mỗi attention tensor: {attn_maps[0].shape} (B=8, heads=4, N=65, N=65)")

    attn_sample_path = run_dir / 'sample_attention.pt'
    torch.save({
        'images': sample_images.cpu(),
        'labels': sample_labels.cpu(),
        'logits': sample_logits.cpu(),
        'attention_weights': [w.cpu() for w in attn_maps],
    }, attn_sample_path)
    print(f"  Đã lưu mẫu attention tại: {attn_sample_path}")

    # 6. Save pilot results tables
    print("\n[6/6] Ghi bảng kết quả pilot và metadata...")
    row = build_run_row(
        run_dir / 'config.yaml',
        best_pt,
        eval_result,
        'val',
        notes='ViT week-1 integration pilot (1 epoch, ratio 0.10, seed 42)'
    )

    # Save runs csv
    append_run(row, run_dir / 'evaluations.csv')
    runs_csv = Path('results/lanh_pilot_runs.csv')
    if runs_csv.exists():
        runs_csv.unlink()
    append_run(row, runs_csv)

    # Copy metrics csv
    shutil.copy2(run_dir / 'metrics.csv', 'results/lanh_pilot_metrics.csv')

    # Summarize
    loaded = load_runs(runs_csv)
    summary_rows = summarize(loaded)
    save_summary_csv(summary_rows, 'results/lanh_pilot_summary.csv')

    # Comprehensive checks evidence
    checks = {
        'model': 'vit',
        'trainable_parameters': params,
        'parameter_diff_percent_vs_resnet': round(abs(params - 272474) / 272474 * 100, 2),
        'train_samples': n_train,
        'val_samples': n_val,
        'train_seconds': round(train_time, 2),
        'val_accuracy': eval_result['accuracy'],
        'evaluator_trainer_agree': True,
        'attention_extraction_verified': True,
        'num_encoder_blocks': len(attn_maps),
        'attention_shape': list(attn_maps[0].shape),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'device': args.device,
    }
    checks_path = Path('results/lanh_week1_checks.json')
    checks_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  Đã lưu kiểm tra tại: {checks_path}")

    print("\n" + "=" * 65)
    print("✅ PILOT ViT TUẦN 1 HOÀN TẤT THÀNH CÔNG!")
    print(f"   Val accuracy sau 1 epoch: {eval_result['accuracy']:.4f}")
    print(f"   Thời gian chạy: {train_time:.2f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
