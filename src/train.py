"""Minimal model-independent trainer. See docs/minh-week1.md for the contract."""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint import load_checkpoint, random_state, restore_random_state, save_checkpoint
from src.models.resnet import count_parameters


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, samples, steps = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect data/config before restarting')
            if training:
                loss.backward()
                optimizer.step()
            size = labels.size(0)
            loss_sum += loss.item() * size
            correct += (logits.argmax(1) == labels).sum().item()
            samples += size
            steps += 1
    if samples == 0:
        raise ValueError('Empty DataLoader')
    return {'loss': loss_sum / samples, 'accuracy': correct / samples,
            'samples': samples, 'steps': steps}


def environment():
    def git(*args):
        return subprocess.run(['git', *args], capture_output=True, text=True, check=False).stdout.strip()

    digest = hashlib.sha256()
    for file in sorted(Path('src').rglob('*.py')):
        digest.update(file.as_posix().encode())
        digest.update(file.read_bytes())
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'platform': platform.platform(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'source_sha256': digest.hexdigest()}


def write_metrics(path, history):
    if history:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def fit(model, train_loader, val_loader, config, run_dir, *, device='cpu',
        resume=None, stop_after_epoch=None):
    """Train to the configured horizon; stop_after_epoch preserves the LR schedule.

    Resume is from last.pt in the same run, at an epoch boundary, with identical
    config and data factories. num_workers=0 is required for this first version.
    """
    device, run_dir = torch.device(device), Path(run_dir)
    settings = config['training']
    epochs = settings['epochs']
    if epochs < 1 or (stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs):
        raise ValueError('Epoch bounds must be positive and within the configured horizon')
    if any(loader.num_workers != 0 for loader in (train_loader, val_loader)):
        raise ValueError('Week-1 deterministic resume requires DataLoader num_workers=0')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a smoke test')
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run already exists: {run_dir}; resume or choose another --run-dir')
    if resume is not None and Path(resume).resolve() != (run_dir / 'last.pt').resolve():
        raise ValueError('Resume must use last.pt from this run directory')
    seed_everything(config['seed'])
    model.to(device)
    optimizer_name = settings.get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=settings['lr'],
            momentum=settings.get('momentum', 0.9), weight_decay=settings.get('weight_decay', 0))
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                     weight_decay=settings.get('weight_decay', 0))
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start, global_step, best, best_epoch, history = 0, 0, -math.inf, 0, []
    if resume:
        state = load_checkpoint(resume)
        if state['config'] != config:
            raise ValueError('Resume config differs; use the original config/horizon')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, global_step = state['epoch'], state['global_step']
        best, best_epoch, history = state['best_val_accuracy'], state['best_epoch'], state['history']
        restore_random_state(state['random'], (train_loader, val_loader))
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        metadata = {**environment(), 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'parameters': count_parameters(model), 'device': str(device), 'config': config}
        (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    end = stop_after_epoch if stop_after_epoch is not None else epochs
    writer = SummaryWriter(str(run_dir / 'tensorboard'), purge_step=start + 1 if resume else None)
    try:
        for epoch in range(start + 1, end + 1):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            started = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train = run_epoch(model, train_loader, device, optimizer)
            sync(device)
            train_seconds = time.perf_counter() - started
            val = run_epoch(model, val_loader, device)
            sync(device)
            global_step += train['steps']
            improved = val['accuracy'] > best
            if improved:
                best, best_epoch = val['accuracy'], epoch
            row = {'epoch': epoch, 'global_step': global_step, 'lr': lr,
                   'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
                   'train_seconds': train_seconds, 'epoch_seconds': time.perf_counter() - started,
                   'peak_cuda_mb': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0}
            history.append(row)
            scheduler.step()
            state = {'epoch': epoch, 'global_step': global_step, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'best_val_accuracy': best, 'best_epoch': best_epoch, 'config': config,
                     'random': random_state((train_loader, val_loader)), 'history': history}
            if improved:
                save_checkpoint(run_dir / 'best.pt', state)
            save_checkpoint(run_dir / 'last.pt', state)
            write_metrics(run_dir / 'metrics.csv', history)
            for key, value in row.items():
                if key != 'epoch':
                    writer.add_scalar(key, value, epoch)
            writer.flush()
            print(json.dumps(row), flush=True)
    finally:
        writer.close()
    return history


def factory(path):
    module, name = path.split(':')
    return getattr(importlib.import_module(module), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed_everything(config['seed'])
    model = factory(config['model']['factory'])(config)
    train_loader, val_loader = factory(config['data']['factory'])(config)
    fit(model, train_loader, val_loader, config, args.run_dir, device=args.device,
        resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == '__main__':
    main()
