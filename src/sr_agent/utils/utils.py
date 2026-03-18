# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import torch
import random
import numpy as np

__all__ = [
    "softmax",
    "seed_all",
]


def softmax(x):
    x = np.exp(x - x.max())
    return x / x.sum()


def seed_all(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
