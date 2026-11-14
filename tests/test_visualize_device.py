import csv
import json

import pytest
import yaml

import src.visualize as visualize


# Hàm test: công cụ vẽ phải dùng thiết bị đã chọn hoặc thiết bị train được ghi trong metadata.
# Giả lập biên evaluator để test chạy được cả trên máy CPU; số liệu ở đây chỉ là fixture đơn vị.
@pytest.mark.parametrize('requested,expected', [('cuda', 'cuda'), (None, 'cuda'), ('cpu', 'cpu')])
def test_visualization_routes_evaluation_to_the_requested_or_training_device(tmp_path, monkeypatch, requested, expected):
    config = {'data': {'split_seed': 2026}, 'model': {'factory': 'src.models.vit:build_model'}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (tmp_path / 'metadata.json').write_text(json.dumps({'device': 'cuda'}), encoding='utf-8')
    with (tmp_path / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_accuracy'])
        writer.writeheader()
        writer.writerow({'epoch': 1, 'val_accuracy': 0.5})
    called = []

    # Hàm giả tại biên tính toán: ghi thiết bị được chuyển xuống evaluator, không giả huấn luyện thật.
    def evaluate(path, config, split='val', device='cpu'):
        called.append(device)
        return {'accuracy': 0.5}

    monkeypatch.setattr(visualize, 'evaluate_checkpoint', evaluate)
    monkeypatch.setattr(visualize, 'build_run_row', lambda *args: {
        'best_epoch': 1, 'checkpoint_epoch': 1, 'commit': 'fixture', 'params': 1,
        'manifest_sha256': 'fixture', 'checkpoint_sha256': 'fixture', 'protocol_id': 'fixture',
        'global_step': 1, 'train_seconds_total': 1.0, 'ratio': '0.10',
    })
    monkeypatch.setattr(visualize, 'load_manifest', lambda *args: {'subsets': {'0.10': [0, 1]}})
    _, _, row, _ = visualize.inspect_run(tmp_path, device=requested)
    assert called == [expected]
    assert row['evaluation_device'] == expected
