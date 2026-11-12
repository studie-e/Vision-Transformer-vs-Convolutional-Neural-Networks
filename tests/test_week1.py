import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import CifarResNet, count_parameters
from src.train import fit, run_epoch, seed_everything


torch.set_num_threads(1)


def test_resnet_classifies_small_rgb_images_and_has_finite_gradients():
    model = CifarResNet(width=8)
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)
    nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert count_parameters(CifarResNet(width=16)) > count_parameters(model)


def test_resnet_can_memorize_a_fixed_batch():
    seed_everything(17)
    model = CifarResNet(width=4, blocks_per_stage=1)
    images, labels = torch.randn(8, 3, 32, 32), torch.arange(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    first_loss = nn.functional.cross_entropy(model(images), labels).item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    assert loss.item() < first_loss / 5


def test_validation_uses_all_samples_and_does_not_change_batchnorm():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    x, y = torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    before = model[0].running_mean.clone()
    model.eval()
    with torch.no_grad():
        expected = nn.functional.cross_entropy(model(x), y).item()
    result = run_epoch(model, loader, torch.device('cpu'))
    assert result['loss'] == pytest.approx(expected)
    assert result['samples'] == 5
    assert torch.equal(model[0].running_mean, before)
    assert all(p.grad is None for p in model.parameters())


def loaders():
    g = torch.Generator().manual_seed(123)
    data = TensorDataset(torch.randn(12, 4, generator=g), torch.arange(12) % 2)
    return (
        DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(17)),
        DataLoader(data, batch_size=5, generator=torch.Generator().manual_seed(18)),
    )


def model_for_resume():
    seed_everything(17)
    return nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 2))


CONFIG = {'seed': 17, 'training': {'epochs': 3, 'lr': 0.05, 'optimizer': 'sgd',
          'momentum': 0.9, 'weight_decay': 0.0}}


def test_resume_matches_uninterrupted_training_including_optimizer_rng_and_schedule(tmp_path):
    full = model_for_resume()
    fit(full, *loaders(), CONFIG, tmp_path / 'full', device='cpu')
    interrupted = model_for_resume()
    fit(interrupted, *loaders(), CONFIG, tmp_path / 'resumed', device='cpu', stop_after_epoch=1)
    resumed = model_for_resume()
    fit(resumed, *loaders(), CONFIG, tmp_path / 'resumed', device='cpu',
        resume=tmp_path / 'resumed' / 'last.pt')
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    a = torch.load(tmp_path / 'full' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=True)
    assert a['global_step'] == b['global_step'] == 9
    assert a['scheduler'] == b['scheduler']
    assert a['best_val_accuracy'] == b['best_val_accuracy']
    rows = list(csv.DictReader((tmp_path / 'resumed' / 'metrics.csv').open()))
    assert [int(r['epoch']) for r in rows] == [1, 2, 3]
    assert (tmp_path / 'resumed' / 'best.pt').exists()
    assert list((tmp_path / 'resumed' / 'tensorboard').glob('events.*'))


def test_trainer_refuses_overwriting_an_existing_run_or_changed_resume_config(tmp_path):
    model = model_for_resume()
    fit(model, *loaders(), CONFIG, tmp_path, device='cpu', stop_after_epoch=1)
    with pytest.raises(FileExistsError):
        fit(model_for_resume(), *loaders(), CONFIG, tmp_path, device='cpu')
    changed = {**CONFIG, 'seed': 42}
    with pytest.raises(ValueError, match='config'):
        fit(model_for_resume(), *loaders(), changed, tmp_path, device='cpu', resume=tmp_path / 'last.pt')


def test_pilot_indices_are_repeatable_disjoint_and_reject_invalid_sizes():
    from src.pilot_data import pilot_indices

    train, val = pilot_indices(100, 40, 20, 2026)
    assert (train, val) == pilot_indices(100, 40, 20, 2026)
    assert len(set(train)) == 40 and len(set(val)) == 20
    assert set(train).isdisjoint(val)
    with pytest.raises(ValueError):
        pilot_indices(100, 90, 20, 2026)
