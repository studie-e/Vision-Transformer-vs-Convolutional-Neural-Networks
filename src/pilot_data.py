"""Week-1 diagnostic data only; replace with Hien's official split factory.

Loads only the CIFAR-10 training partition. No test data is used for validation.
Source: https://docs.pytorch.org/vision/0.26/generated/torchvision.datasets.CIFAR10.html
"""

import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url


def pilot_indices(size, train_samples, val_samples, seed):
    if min(train_samples, val_samples) < 1 or train_samples + val_samples > size:
        raise ValueError('Pilot sizes must be positive, disjoint and fit the training partition')
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[:train_samples], indices[train_samples:train_samples + val_samples]


def build_loaders(config):
    data = config['data']
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    if data.get('source', 'torchvision') == 'fastai':
        root = Path(data['root'])
        train_dir = root / 'cifar10' / 'train'
        if not train_dir.exists() and data.get('download', False):
            download_url('https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz', str(root),
                         filename='cifar10-fastai.tgz', md5='7d7b892d003476893d94d030e4f7c0f0')
            with tarfile.open(root / 'cifar10-fastai.tgz') as archive:
                archive.extractall(root, filter='data')
        dataset = datasets.ImageFolder(str(train_dir), transform=transform)
        if len(dataset) != 50000 or len(dataset.classes) != 10:
            raise ValueError('Expected 50,000 training images across 10 CIFAR classes')
    elif data.get('source', 'torchvision') == 'torchvision':
        dataset = datasets.CIFAR10(root=data['root'], train=True, transform=transform,
                                   download=data.get('download', False))
    else:
        raise ValueError('Pilot source must be torchvision or fastai')
    train_indices, val_indices = pilot_indices(
        len(dataset), data['train_samples'], data['val_samples'], data['split_seed'])
    return (
        DataLoader(Subset(dataset, train_indices), batch_size=data['batch_size'],
                   shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(config['seed'])),
        DataLoader(Subset(dataset, val_indices), batch_size=data['batch_size'],
                   shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(config['seed'] + 1)),
    )
