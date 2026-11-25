# Người phụ trách: Lan Anh (kiểm thử tự động cấu hình và kết quả ViT/Ablation tuần 3–4).
"""Kiểm thử tự động cho các cấu hình và lượt chạy GPU thực tế của Lan Anh:
- Tính hợp lệ của 6 YAML ViT tuần 3 (T13).
- Cấu hình và kiến trúc Ablation bỏ Positional Embedding (T16, 268.042 tham số).
- Tính toàn vẹn của các log GPU thực tế (số ảnh đúng nguyên, thời gian dương, đủ 100 epoch).
- Thống kê 3-seed thực tế của ViT ở mức 10% dữ liệu (mean ± std với ddof=1).
"""

import csv
import math
from pathlib import Path

import pytest
import yaml

from scripts.reproduce_week4 import check_history
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

        model = build_model(cfg)
        params = count_parameters(model)
        assert params == 268042, f'Số tham số ablation {params} khác 268.042'
        assert 'pos_embed' not in [n for n, _ in model.named_parameters()]
        assert 'pos_embed' in [n for n, _ in model.named_buffers()]

    baseline_vit = VisionTransformer(embed_dim=96, depth=4, num_heads=4, mlp_ratio=1.5, use_pos_embed=True)
    assert count_parameters(baseline_vit) == 274282
    assert 274282 - 268042 == 6240


def test_real_vit_10_gpu_logs_pass_strict_audit():
    with Path('results/week3/vit_runs.csv').open(encoding='utf-8', newline='') as f:
        run_rows = {r['run_id']: r for r in csv.DictReader(f)}

    assert set(run_rows) >= {'vit_10_seed43', 'vit_10_seed44'}
    for run_id in ('vit_10_seed43', 'vit_10_seed44'):
        metrics_path = Path(f'results/week3/{run_id}_metrics.csv')
        assert metrics_path.is_file(), f'Thiếu metrics thật cho {run_id}'
        with metrics_path.open(encoding='utf-8', newline='') as f:
            history = list(csv.DictReader(f))
        assert check_history(history, run_rows[run_id], epochs=100) == []


def test_vit_summary_statistics_reflect_verified_runs():
    summary_path = Path('results/week3/vit_summary.csv')
    assert summary_path.is_file(), f'Thiếu {summary_path}'

    with summary_path.open(encoding='utf-8', newline='') as f:
        ratios = {r['ratio']: r for r in csv.DictReader(f)}

    assert set(ratios) == {'0.10', '0.50', '1.00'}

    # 10% đã có đủ 3 seed thực tế: [0.5216, 0.5398, 0.5182]
    r10 = ratios['0.10']
    assert int(r10['n_seeds']) == 3
    assert float(r10['mean_accuracy']) == pytest.approx((0.5216 + 0.5398 + 0.5182) / 3, abs=1e-6)
    assert float(r10['std_accuracy']) == pytest.approx(0.0116143589, abs=1e-6)

    # 50% và 100% hiện giữ 1 seed đã xác minh (seed 42), chờ 4 lượt RTX 5080
    for ratio_key, expected_acc in (('0.50', 0.6766), ('1.00', 0.7590)):
        row = ratios[ratio_key]
        assert int(row['n_seeds']) == 1
        assert float(row['mean_accuracy']) == pytest.approx(expected_acc, abs=1e-6)
        assert math.isnan(float(row['std_accuracy']))
