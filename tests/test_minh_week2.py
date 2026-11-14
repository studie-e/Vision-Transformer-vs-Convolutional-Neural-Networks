"""Kiểm tra giao thức tuần 2 và prototype Swin, không thay cho thí nghiệm thật."""

import copy

import pytest
import torch

from scripts.run_minh_week2 import choose_recipe, make_tuning_configs
from src.models.resnet import count_parameters
from src.models.swin import build_model


# Hàm test: hai kiến trúc phải được thử cùng số recipe/epoch và cùng split, không ưu ái CNN.
# tmp_path do pytest tạo để các YAML kiểm tra không ghi vào cấu hình thật của nhóm.
def test_tuning_budget_is_equal_and_uses_the_main_manifest(tmp_path):
    paths = make_tuning_configs(tmp_path)
    import yaml
    configs = [yaml.safe_load(path.read_text(encoding='utf-8')) for path in paths]
    assert len(configs) == 4
    for model in ('resnet', 'vit'):
        selected = [c for c in configs if model in c['model']['factory']]
        assert len(selected) == 2
        assert {c['training']['optimizer'] for c in selected} == {'sgd', 'adamw'}
        assert {c['training']['epochs'] for c in selected} == {10}
    assert all(c['seed'] == 42 and c['data']['split_seed'] == 2026 for c in configs)
    assert all(c['data']['ratio'] == '0.50' for c in configs)
    assert all(c['data']['factory'] == 'src.data:build_loaders' for c in configs)


# Hàm test: chọn recipe bằng validation, giữ thứ tự khi hòa và không chọn riêng lượt chạy dở.
def test_recipe_selection_uses_validation_and_refuses_incomplete_candidates():
    rows = [
        {'recipe': 'sgd', 'best_val_accuracy': 0.6, 'completed_epochs': 10},
        {'recipe': 'adamw', 'best_val_accuracy': 0.7, 'completed_epochs': 10},
    ]
    assert choose_recipe(rows)['recipe'] == 'adamw'
    tied = copy.deepcopy(rows)
    tied[1]['best_val_accuracy'] = 0.6
    assert choose_recipe(tied)['recipe'] == 'sgd'
    rows[1]['completed_epochs'] = 2
    with pytest.raises(ValueError, match='epoch'):
        choose_recipe(rows)


# Hàm test: Swin thư viện được cấu hình cho ảnh nhỏ, logits và gradient đều đúng.
# Không train Swin tuần 2; chỉ xác nhận prototype sẵn sàng cho phần mở rộng tuần 3.
def test_swin_prototype_accepts_32_pixel_rgb_and_has_finite_gradients():
    torch.set_num_threads(1)
    model = build_model({'model': {'kwargs': {}}})
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert abs(count_parameters(model) - 272474) / max(count_parameters(model), 272474) <= 0.10


# Hàm test: YAML đã khóa phải được giữ nguyên, kể cả thay ít epoch để chạy nhanh cũng bị từ chối.
# monkeypatch chỉ đổi thư mục làm việc sang fixture; không đụng giao thức thực đang huấn luyện.
def test_locked_configs_reject_reducing_the_training_budget(tmp_path, monkeypatch):
    import json
    import yaml
    from scripts.run_minh_week2 import RATIOS, locked_resnet_configs

    root = tmp_path / 'configs/week2'
    root.mkdir(parents=True)
    locked = {'configs': {}}
    for label in RATIOS:
        name = f'resnet_{label}_s42.yaml'
        config = {'training': {'epochs': 100}, 'data': {'ratio': RATIOS[label]}}
        locked['configs'][name] = copy.deepcopy(config)
        (root / name).write_text(yaml.safe_dump(config), encoding='utf-8')
    (root / 'locked_protocol.json').write_text(json.dumps(locked), encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert len(locked_resnet_configs()) == 3
    changed = locked['configs']['resnet_d10_s42.yaml']
    changed['training']['epochs'] = 1
    (root / 'resnet_d10_s42.yaml').write_text(yaml.safe_dump(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='after locking'):
        locked_resnet_configs()
