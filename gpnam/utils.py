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

def process_in_chunks_regressor(function, *args, batch_size, out=None, **kwargs):
    """Computes output by applying batch-parallel function to large data tensor in chunks.

    Args:
        function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...].
        args: one or many tensors, each [num_instances, ...].
        batch_size: maximum chunk size processed in one go.
        out: memory buffer for out, defaults to torch.zeros of appropriate size and type.

    Returns:
        out: the outputs of function(data), computed in a memory-efficient (mini-batch) way.
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])[0]
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])[0]
    return out

def process_in_chunks_classifier(function, *args, batch_size, out=None, **kwargs):
    """Computes output by applying batch-parallel function to large data tensor in chunks.

    Args:
        function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...].
        args: one or many tensors, each [num_instances, ...].
        batch_size: maximum chunk size processed in one go.
        out: memory buffer for out, defaults to torch.zeros of appropriate size and type.

    Returns:
        out: the outputs of function(data), computed in a memory-efficient (mini-batch) way.
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out

def check_numpy(x):
    """Makes sure x is a numpy array. If not, make it as one."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x

def sigmoid(x):
    """A sigmoid function for numpy array.

    Args:
        x: numpy array.

    Returns:
        the sigmoid value.
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


