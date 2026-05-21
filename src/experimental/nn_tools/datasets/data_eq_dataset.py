# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
# Minimal usage:

# Remember to add "src/experimental/" to PYTHONPATH.
from torch.utils.data import DataLoader
from nn_tools.datasets.data_eq_dataset import DataEqDataset
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.models import EquationEmbedder

eq_generator = BaseEqGenerator.create("gplearn", n_variables=2)
data_generator = BaseDataGenerator.create("uniform", sample_num=100)
equation_embedder = EquationEmbedder(d_model=128, max_variables=2, operands=eq_generator.symbols)
dataset = DataEqDataset(
    2,
    eq_generator,
    data_generator,
    n_samples=None,
    equation_embedder=equation_embedder,
)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=dataset.get_sampler(),
    collate_fn=dataset.collate_fn,
)
for item in dataloader: # 可以无限迭代生成数据
    print(
        item["formula"], 
        item["data"].shape, 
        item["token"],
        item["index"].shape
    )
    break
"""
from __future__ import annotations

import torch
import logging
import warnings
import itertools
import numpy as np
import torch.utils.data as D
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from ..models import EquationEmbedder
from .generate_eq import BaseEqGenerator
from .generate_data import BaseDataGenerator

# 屏蔽 eval 结果赋值时的 overflow 警告
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

_logger = logging.getLogger(f"sr_agent.{__name__}")
__all__ = ["DataEqDataset", "InfiniteSampler"]

class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()


class DataEqDataset(D.Dataset):
    """Generate scalar data/formula pairs.

    Each item contains:
    - ``data``: float tensor with shape ``(sample_num, max_var_num + 1)``.
      The first columns are variables in sorted variable-name order, padded
      with zeros when the equation uses fewer than ``max_var_num`` variables.
      The final column is the target value.
    - ``formula``: the generated equation.
    - ``token`` and ``index``: tokenized/vectorized formula, returned only
      when ``equation_embedder`` is provided.
    """

    def __init__(
        self,
        max_var_num: int,
        eq_generator: BaseEqGenerator,
        data_generator: BaseDataGenerator,
        n_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        equation_embedder: Optional[EquationEmbedder] = None,
    ):
        if max_var_num <= 0:
            raise ValueError("max_var_num must be positive.")
        self.max_var_num = max_var_num
        self.eq_generator = eq_generator
        self.data_generator = data_generator
        self.n_samples = n_samples
        self.random_state = random_state
        self.equation_embedder = equation_embedder

    def __len__(self):
        # 如果 n_samples 为 None, 实际的无限循环由 InfiniteSampler 接管
        if self.n_samples is None:
            raise TypeError("Infinite DataEqDataset has no length; use get_sampler() with DataLoader.")
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.random_state, idx)) if self.random_state is not None else None
        eqtree = self.eq_generator(_rng=rng)
        data_dict, target, success = self.data_generator(eqtree, _rng=rng)

        variables = sorted(data_dict)
        if len(variables) > self.max_var_num:
            raise ValueError(
                f"Equation uses {len(variables)} variables, but max_var_num={self.max_var_num}."
            )

        sample_num = int(target.shape[0])
        data = np.zeros((sample_num, self.max_var_num + 1), dtype=np.float32)
        for i, variable in enumerate(variables):
            data[:, i] = np.asarray(data_dict[variable], dtype=np.float32).reshape(-1)
        data[:, -1] = np.asarray(target, dtype=np.float32).reshape(-1)

        if not success:
            _logger.warning("Generated data for formula %s contains invalid samples.", eqtree)

        item = {
            "data": torch.from_numpy(data),
            "formula": eqtree,
        }
        if self.equation_embedder is not None:
            token = self.equation_embedder.tokenize(eqtree)
            index = self.equation_embedder.vectorize(token, variables)
            item["token"] = token
            item["index"] = torch.as_tensor(index, dtype=torch.long)
        return item

    def collate_fn(self, batch):
        result = {
            "data": torch.stack([item["data"] for item in batch], dim=0),
            "formula": [item["formula"] for item in batch],
        }
        if "token" in batch[0]:
            pad_token = self.equation_embedder.PAD_TOKEN
            result["token"] = [
                item["token"] + [pad_token] * (self.max_var_num - len(item["token"]))
                for item in batch
            ]
        if "index" in batch[0]:
            pad_index = self.equation_embedder.pad_token_id
            result["index"] = pad_sequence(
                [item["index"] for item in batch],
                batch_first=True, padding_value=pad_index,
            )
        return result

    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None
