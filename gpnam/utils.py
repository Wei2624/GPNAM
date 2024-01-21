import torch
import numpy as np
import os
from logging import log
import random

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        return {'x': torch.from_numpy(x),
                'y': torch.from_numpy(y)}


def seed_everything(seed=None) -> int:
    """Seed everything.

    It includes pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable. Borrow
    it from the pytorch_lightning.

    Args:
        seed: the seed. If None, it generates one.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    print(f"No correct seed found, seed set to {seed}")
    return seed



