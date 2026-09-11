# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import random
import builtins

__all__ = [
    "softmax",
    "seed_all",
    "bounded_value",
]


def bounded_value(value, min, max, default, converter=None):
    """Convert a value when requested, then constrain it to an inclusive range."""
    try:
        parsed = converter(value) if converter is not None else value
    except (TypeError, ValueError):
        parsed = default
    return builtins.max(min, builtins.min(max, parsed))


def softmax(x):
    import numpy as np
    x = np.exp(x - x.max())
    return x / x.sum()


def seed_all(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
