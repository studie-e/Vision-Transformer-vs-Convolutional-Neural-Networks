# Người phụ trách: Lan Anh (kiểm thử tự động tuần 3).
"""Kiểm thử tự động cho các sản phẩm tuần 3 của Lan Anh:
- Tính hợp lệ của 6 YAML ViT tuần 3 (T13).
- Cấu hình và kiến trúc Ablation bỏ Positional Embedding (T16).
- Số tham số ViT ablation (268.042) chênh lệch đúng 6.240 so với baseline (274.282).
- Tính toàn vẹn của 100 epoch metrics và quy tắc chọn best checkpoint.
- Thống kê 3-seed của ViT chính và Ablation (mean ± std với ddof=1).
"""

import statistics
from pathlib import Path

import pytest
import yaml

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


# Lan Anh đối chiếu bảng kết quả với số đếm từ log và thống kê seed, kể cả khi ablation tăng điểm.
@pytest.mark.parametrize('kind,ratios', [('vit', ('0.10', '0.50', '1.00')), ('ablation', ('0.50',))])
def test_week3_measurements_and_seed_summary(kind, ratios):
    from scripts.reproduce_week4 import check_history, load_runs

    folder = Path('results/week3')
    repeats = load_runs(folder / f'{kind}_runs.csv')
    expected_seeds = {43, 44} if kind == 'vit' else {42, 43, 44}
    assert len(repeats) == len(ratios) * len(expected_seeds)
    assert {(row['ratio'], int(row['train_seed'])) for row in repeats} == {
        (ratio, seed) for ratio in ratios for seed in expected_seeds}
    for row in repeats:
        assert int(row['split_seed']) == int(row['train_seed']) + 1984
        history = load_runs(folder / f"{row['run_id']}_metrics.csv")
        assert check_history(history, row) == [], row['run_id']
    rows = list(repeats)
    if kind == 'vit':
        rows.extend(row for row in load_runs(Path('results/team_week2/p0_runs.csv'))
                    if row['model'] == 'src.models.vit:build_model')
    summaries = load_runs(folder / f'{kind}_summary.csv')
    assert sorted(s['ratio'] for s in summaries) == list(ratios)
    for summary in summaries:
        group = [float(row['accuracy']) for row in rows if row['ratio'] == summary['ratio']]
        assert int(summary['n_seeds']) == len(group) == 3
        assert float(summary['mean_accuracy']) == pytest.approx(statistics.mean(group))
        assert float(summary['std_accuracy']) == pytest.approx(statistics.stdev(group))
