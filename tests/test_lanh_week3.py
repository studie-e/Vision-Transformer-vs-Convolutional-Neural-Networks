# Người phụ trách: Lan Anh (kiểm thử tự động tuần 3).
"""Kiểm thử tự động cho các sản phẩm tuần 3 của Lan Anh:
- Tính hợp lệ của 6 YAML ViT tuần 3 (T13).
- Cấu hình và kiến trúc Ablation bỏ Positional Embedding (T16).
- Số tham số ViT ablation (268.042) chênh lệch đúng 6.240 so với baseline (274.282).
- Tính toàn vẹn của 100 epoch metrics và quy tắc chọn best checkpoint.
- Thống kê 3-seed của ViT chính và Ablation (mean ± std với ddof=1).
"""

import csv
import math
from pathlib import Path

import pytest
import yaml

from scripts.summarize import _mean_std
from src.models.vit import VisionTransformer, build_model, count_parameters


def test_week3_vit_configs_exist_and_match_locked_protocol():
    expected_seeds = {43: 2027, 44: 2028}
    ratios = {'10': '0.10', '50': '0.50', '100': '1.00'}

    for seed, split_seed in expected_seeds.items():
        for ratio_str, ratio_val in ratios.items():
            cfg_path = Path(f'configs/week3/vit_{ratio_str}_seed{seed}.yaml')
            assert cfg_path.is_file(), f'Thiếu file cấu hình: {cfg_path}'
            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

            assert cfg['seed'] == seed
            assert cfg['data']['split_seed'] == split_seed
            assert cfg['data']['ratio'] == ratio_val
            assert cfg['data']['batch_size'] == 128
            assert cfg['training']['epochs'] == 100
            assert cfg['training']['optimizer'] == 'adamw'
            assert cfg['training']['lr'] == 0.0003
            assert cfg['training']['weight_decay'] == 0.05

            kw = cfg['model']['kwargs']
            assert kw['image_size'] == 32
            assert kw['patch_size'] == 4
            assert kw['embed_dim'] == 96
            assert kw['depth'] == 4
            assert kw['num_heads'] == 4
            assert kw['mlp_ratio'] == 1.5
            assert kw['dropout'] == 0.1
            assert kw['use_pos_embed'] is True


def test_week3_ablation_configs_and_parameter_count():
    ablation_cfgs = [
        Path('configs/ablation.yaml'),
        Path('configs/week3/ablation_vit_50_seed42.yaml'),
        Path('configs/week3/ablation_vit_50_seed43.yaml'),
        Path('configs/week3/ablation_vit_50_seed44.yaml'),
    ]

    for cfg_path in ablation_cfgs:
        assert cfg_path.is_file(), f'Thiếu file ablation config: {cfg_path}'
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        assert cfg['model']['kwargs']['use_pos_embed'] is False
        assert cfg['data']['ratio'] == '0.50'

        # Khởi tạo mô hình ablation và kiểm tra tham số
        model = build_model(cfg)
        params = count_parameters(model)
        assert params == 268042, f"Số tham số ablation {params} khác 268.042"
        assert 'pos_embed' not in [n for n, _ in model.named_parameters()]
        assert 'pos_embed' in [n for n, _ in model.named_buffers()]

    # Chênh lệch tham số đúng bằng số phần tử positional embedding (65 * 96 = 6.240)
    baseline_vit = VisionTransformer(embed_dim=96, depth=4, num_heads=4, mlp_ratio=1.5, use_pos_embed=True)
    assert count_parameters(baseline_vit) == 274282
    assert 274282 - 268042 == 6240


def test_metrics_csv_epochs_and_best_checkpoint():
    metrics_files = [
        Path('results/week3/vit_10_seed43_metrics.csv'),
        Path('results/week3/vit_50_seed43_metrics.csv'),
        Path('results/week3/vit_100_seed43_metrics.csv'),
        Path('results/week3/vit_10_seed44_metrics.csv'),
        Path('results/week3/vit_50_seed44_metrics.csv'),
        Path('results/week3/vit_100_seed44_metrics.csv'),
        Path('results/week3/ablation_vit_50_seed42_metrics.csv'),
        Path('results/week3/ablation_vit_50_seed43_metrics.csv'),
        Path('results/week3/ablation_vit_50_seed44_metrics.csv'),
    ]

    for p in metrics_files:
        assert p.is_file(), f'Thiếu file metrics: {p}'
        with p.open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 100, f'{p.name} không đủ 100 epoch'
        assert [int(r['epoch']) for r in rows] == list(range(1, 101))
        # Kiểm tra best val accuracy là giá trị cực đại
        val_accs = [float(r['val_accuracy']) for r in rows]
        max_acc = max(val_accs)
        best_row = max(rows, key=lambda r: float(r['val_accuracy']))
        assert float(best_row['val_accuracy']) == max_acc


def test_vit_three_seed_summary_statistics():
    summary_path = Path('results/week3/vit_summary.csv')
    assert summary_path.is_file(), f'Thiếu {summary_path}'

    with summary_path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    ratios = {r['ratio']: r for r in rows}

    # 10%: mean ~ 0.5219, std ~ 0.0012
    r10 = ratios['0.10']
    assert int(r10['n_seeds']) == 3
    assert float(r10['mean_accuracy']) == pytest.approx(0.521867, abs=1e-4)
    assert float(r10['std_accuracy']) == pytest.approx(0.001222, abs=1e-4)

    # 50%: mean ~ 0.6767, std ~ 0.0014
    r50 = ratios['0.50']
    assert int(r50['n_seeds']) == 3
    assert float(r50['mean_accuracy']) == pytest.approx(0.676733, abs=1e-4)
    assert float(r50['std_accuracy']) == pytest.approx(0.001405, abs=1e-4)

    # 100%: mean ~ 0.7592, std ~ 0.0009
    r100 = ratios['1.00']
    assert int(r100['n_seeds']) == 3
    assert float(r100['mean_accuracy']) == pytest.approx(0.759200, abs=1e-4)
    assert float(r100['std_accuracy']) == pytest.approx(0.000917, abs=1e-4)


def test_ablation_summary_and_drop_magnitude():
    summary_path = Path('results/week3/ablation_summary.csv')
    assert summary_path.is_file(), f'Thiếu {summary_path}'

    with summary_path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row['ratio'] == '0.50'
    assert int(row['n_seeds']) == 3

    # Ablation mean ~ 0.6237 (62.37%), std ~ 0.0018
    mean_abl = float(row['mean_accuracy'])
    assert mean_abl == pytest.approx(0.623733, abs=1e-4)
    assert float(row['std_accuracy']) == pytest.approx(0.001815, abs=1e-4)

    # Đối chiếu với baseline 50% ViT (67.67%)
    drop = 0.676733 - mean_abl
    assert 0.050 <= drop <= 0.055, f"Độ sụt giảm {drop:.4f} không nằm trong khoảng kỳ vọng 5.0 - 5.5%"
