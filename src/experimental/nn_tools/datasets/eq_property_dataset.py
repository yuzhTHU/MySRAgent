# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
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
from .data_eq_dataset import InfiniteSampler


__all__ = ["EqPropertyDataset", "InfiniteSampler"]
_logger = logging.getLogger(f"sr_agent.{__name__}")
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")


class EqPropertyDataset(D.Dataset):
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
            raise TypeError("Infinite EqPropertyDataset has no length; use get_sampler() with DataLoader.")
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.random_state, idx)) if self.random_state is not None else None
        while True:
            eqtree = self.eq_generator(_rng=rng)
            data_dict, target, success = self.data_generator(eqtree, _rng=rng)
            if success: break

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

        padded_variables = variables + [f'<VAR:{i}>' for i in range(len(variables), self.max_var_num)]
        properties = {}
        # {'name': 'exists', 'scope': 'variable', 'task': 'binary'} 变量是否存在
        properties['exists'] = [exists(eqtree, var, data_dict) for var in padded_variables]
        # {'name': 'multiple_splitable', 'scope': 'variable_pair', 'task': 'binary'} 变量对是否可被乘法分解
        properties['multiple_splitable'] = [multiple_splitable(eqtree, var1, var2, data_dict) for var1 in padded_variables for var2 in padded_variables]
        # {'name': 'target_mean', 'scope': 'target', 'task': 'regression'} 目标值均值
        properties['target_mean'] = [target.mean().item()]

        item = {"data": torch.from_numpy(data), "formula": eqtree, "properties": properties}
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
            "properties": [item["properties"] for item in batch],
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

def exists(eqtree: nd.Symbol, var: str, data_dict: Dict[str, np.ndarray]):
    for node in eqtree.iter_preorder():
        if type(node).__name__ == 'Variable' and node.name == var:
            return True
    return False

def multiple_splitable(eqtree: nd.Symbol, var1: str, var2: str, data_dict: Dict[str, np.ndarray]):
    return False # 待实现
