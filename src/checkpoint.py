"""Epoch-boundary checkpoints using state_dict, including random generators.

Sources: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
https://docs.pytorch.org/docs/2.11/notes/randomness.html
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def random_state(loaders):
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'loaders': [loader.generator.get_state() if loader.generator else None for loader in loaders],
    }


def restore_random_state(state, loaders):
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    for loader, saved in zip(loaders, state['loaders'], strict=True):
        if (loader.generator is None) != (saved is None):
            raise ValueError('DataLoader generator differs from checkpoint')
        if saved is not None:
            loader.generator.set_state(saved.cpu())


def save_checkpoint(path: Path, state: dict):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)
