"""Kiểm thử tự động cho các module tuần 1 của Hiền:
- src.data: tính phân tầng, không giao nhau, tính lồng nhau, build_loaders.
- src.evaluate: compute_accuracy với ví dụ biết đáp án.
- scripts.summarize: _mean_std với ví dụ biết đáp án.
"""

import math
from pathlib import Path

import pytest
import torch

from scripts.summarize import _mean_std
from src.data import (
    CLASSES, NUM_CLASSES,
    load_manifest, make_eval_transform, make_train_transform,
)
from src.evaluate import compute_accuracy


def sample_row(**changes):
    row = dict(model='resnet', ratio='0.10', split='val', protocol_id='recipe',
               train_seed='42', split_seed='2026', params='272474', accuracy='0.8')
    return {**row, **changes}


def test_summary_rejects_duplicates_and_separates_protocols_and_splits():
    from scripts.summarize import summarize
    with pytest.raises(ValueError, match='Duplicate'):
        summarize([sample_row(), sample_row()])
    result = summarize([sample_row(), sample_row(split='test', accuracy='0.4'),
                        sample_row(protocol_id='other', accuracy='0.2')])
    assert len(result) == 3
    assert sorted(r['mean_accuracy'] for r in result) == [0.2, 0.4, 0.8]
    assert all(r['n_seeds'] == 1 for r in result)


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1'])
def test_summary_rejects_invalid_accuracy(value):
    from scripts.summarize import summarize
    with pytest.raises(ValueError):
        summarize([sample_row(accuracy=value)])


def test_missing_and_corrupted_manifests_fail(tmp_path):
    from src.data import _verify_splits, validate_manifest
    assert not _verify_splits(tmp_path)
    manifest = load_manifest(2026)
    manifest['val_indices'][0] = manifest['train_indices'][0]
    with pytest.raises(ValueError, match='overlap'):
        validate_manifest(manifest)


def test_evaluation_rejects_changed_config_and_protocol_tracks_recipe():
    from src.evaluate import checked_config, protocol_id
    import copy
    import yaml
    config = yaml.safe_load(Path('configs/hien_smoke.yaml').read_text())
    other = copy.deepcopy(config)
    other['data']['split_seed'] = 2027
    other['seed'] = 43
    with pytest.raises(ValueError, match='differs'):
        checked_config({'config': config}, other)
    assert protocol_id(config) == protocol_id(other)
    other['training']['lr'] = 0.1
    assert protocol_id(config) != protocol_id(other)


def test_manifest_fingerprint_rejects_changes():
    import yaml
    from src.data import manifest_digest
    from src.evaluate import checked_config
    config = yaml.safe_load(Path('configs/hien_smoke.yaml').read_text())
    state = {'config': config, 'manifest_sha256': manifest_digest(load_manifest(2026))}
    assert checked_config(state, config) == config
    state['manifest_sha256'] = 'changed'
    with pytest.raises(ValueError, match='fingerprint'):
        checked_config(state, config)


def test_resume_rejects_changed_manifest(tmp_path):
    from src.train import fit
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1]))
    train = DataLoader(dataset, batch_size=2)
    val = DataLoader(dataset, batch_size=2)
    train.manifest_sha256 = 'original'
    config = {'seed': 42, 'training': {'epochs': 2, 'lr': 0.01, 'optimizer': 'sgd'}}
    fit(torch.nn.Linear(2, 2), train, val, config, tmp_path, stop_after_epoch=1)
    train.manifest_sha256 = 'changed'
    with pytest.raises(ValueError, match='manifest'):
        fit(torch.nn.Linear(2, 2), train, val, config, tmp_path, resume=tmp_path / 'last.pt')


def test_checkpoint_evaluation_metadata_csv_and_summary(tmp_path, monkeypatch):
    """Known-answer end-to-end evaluation, without treating fake data as evidence."""
    import json
    import yaml
    import src.data as data
    import src.models.resnet as resnet
    from src.evaluate import evaluate_checkpoint, build_run_row, append_run
    from scripts.summarize import load_runs, summarize

    class KnownDataset:
        def __init__(self, **kwargs):
            self.targets = [i % 10 for i in range(50000)]
        def __len__(self):
            return 50000
        def __getitem__(self, idx):
            label = self.targets[idx]
            return torch.nn.functional.one_hot(torch.tensor(label), 10).float(), label

    monkeypatch.setattr(data.datasets, 'CIFAR10', KnownDataset)
    monkeypatch.setattr(resnet, 'build_model', lambda config: torch.nn.Identity())
    split = data.make_splits(2026)
    data.save_manifest(split, tmp_path / 'splits')
    config = yaml.safe_load(Path('configs/hien_smoke.yaml').read_text())
    config['data']['splits_dir'] = str(tmp_path / 'splits')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / 'best.pt'
    torch.save(dict(model={}, config=config, manifest_sha256=data.manifest_digest(split),
                    history=[{'train_seconds': 1.0}], best_val_accuracy=1.0,
                    best_epoch=1, global_step=36), checkpoint)
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(config=config, commit='test',
        parameters=0, source_sha256='test-source', git_status='test-fixture')))
    result = evaluate_checkpoint(checkpoint, config)
    assert result['correct'] == result['total'] == 5000
    assert len(result['probabilities']) == 5000
    assert all(abs(sum(p) - 1) < 1e-6 for p in result['probabilities'])
    row = build_run_row(config_path, checkpoint, result, 'val')
    output = tmp_path / 'runs.csv'
    append_run(row, output)
    with pytest.raises(ValueError, match='Duplicate'):
        append_run(row, output)
    summary = summarize(load_runs(output, 'val'))
    assert summary[0]['n_seeds'] == 1
    assert summary[0]['mean_accuracy'] == 1.0
    assert math.isnan(summary[0]['std_accuracy'])


def test_real_cifar_loaders_and_reproducible_splits():
    """Integration check when the official dataset is present; never downloads in pytest."""
    import yaml
    from src.data import build_loaders, build_test_loader, make_splits, validate_manifest
    if not Path('data/cifar-10-batches-py/data_batch_1').exists():
        pytest.skip('Official CIFAR-10 not downloaded')
    config = yaml.safe_load(Path('configs/hien_smoke.yaml').read_text())
    train, val = build_loaders(config)
    assert len(build_test_loader(config).dataset) == 10000
    assert len(train.dataset) == 4500 and len(val.dataset) == 5000
    images, labels = next(iter(train))
    assert images.shape == (128, 3, 32, 32)
    assert labels.dtype == torch.int64 and labels.min() >= 0 and labels.max() < 10
    assert torch.equal(val.dataset[0][0], val.dataset[0][0])
    targets = train.dataset.dataset.targets
    for seed in (2026, 2027, 2028):
        manifest = load_manifest(seed)
        validate_manifest(manifest, targets, seed)
        regenerated = make_splits(seed)
        for key in ('train_indices', 'val_indices', 'subsets'):
            assert manifest[key] == regenerated[key]


# ---------------------------------------------------------------------------
# Test evaluate & summarize (không cần dữ liệu thật)
# ---------------------------------------------------------------------------

def test_compute_accuracy_with_known_examples():
    """Kiểm tra tính accuracy bằng ví dụ biết đáp án."""
    # 3/5 đúng = 0.6
    res = compute_accuracy([0, 1, 2, 3, 4], [0, 1, 2, 9, 9])
    assert res['correct'] == 3
    assert res['total'] == 5
    assert abs(res['accuracy'] - 0.6) < 1e-9

    # 100% đúng
    assert compute_accuracy([1, 2, 3], [1, 2, 3])['accuracy'] == 1.0

    # 0% đúng
    assert compute_accuracy([0, 0], [1, 1])['accuracy'] == 0.0

    # Lỗi kích thước không khớp
    with pytest.raises(ValueError):
        compute_accuracy([1, 2], [1])

    # Danh sách rỗng
    with pytest.raises(ValueError):
        compute_accuracy([], [])


def test_mean_std_with_known_examples():
    """Kiểm tra tính mean và std mẫu bằng ví dụ tính tay."""
    # 3 seed: 0.70, 0.72, 0.74 => mean=0.72, std=0.02
    mean, std = _mean_std([0.70, 0.72, 0.74])
    assert abs(mean - 0.72) < 1e-10
    assert abs(std - 0.02) < 1e-10

    # 1 seed => std = NaN
    m1, s1 = _mean_std([0.85])
    assert m1 == 0.85
    assert math.isnan(s1)

    # Rỗng
    m0, s0 = _mean_std([])
    assert math.isnan(m0) and math.isnan(s0)


# ---------------------------------------------------------------------------
# Test data manifests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [2026, 2027, 2028])
def test_manifest_integrity(seed):
    """Kiểm tra tính toàn vẹn của manifest đã lưu trên disk."""
    splits_dir = Path('results/splits')
    manifest_path = splits_dir / f'split_seed{seed}_manifest.json'
    assert manifest_path.exists(), f'Chưa có {manifest_path}'

    m = load_manifest(seed, splits_dir=splits_dir)
    train_idx = m['train_indices']
    val_idx   = m['val_indices']
    subsets   = m['subsets']

    # Đúng số lượng
    assert len(train_idx) == 45_000
    assert len(val_idx) == 5_000

    # Không giao nhau
    assert set(train_idx).isdisjoint(set(val_idx))

    # Tính lồng nhau
    sub10 = set(subsets['0.10'])
    sub50 = set(subsets['0.50'])
    sub100 = set(subsets['1.00'])

    assert len(sub10) == 4_500
    assert len(sub50) == 22_500
    assert len(sub100) == 45_000

    assert sub10.issubset(sub50)
    assert sub50.issubset(sub100)
    assert sub100 == set(train_idx)


def test_transforms_produce_correct_shapes():
    """Kiểm tra transforms cho ra tensor 3x32x32."""
    from PIL import Image

    # Ảnh PIL RGB 32x32
    img = Image.new('RGB', (32, 32), color=(128, 128, 128))

    train_t = make_train_transform()
    eval_t  = make_eval_transform()

    t_tensor = train_t(img)
    e_tensor = eval_t(img)

    assert t_tensor.shape == (3, 32, 32)
    assert e_tensor.shape == (3, 32, 32)
    assert t_tensor.dtype == torch.float32
    assert e_tensor.dtype == torch.float32

