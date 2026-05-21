# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import logging
from collections import defaultdict
from typing import Sequence

import numpy as np
import nd2py as nd

from .base_eq_generator import BaseEqGenerator

_logger = logging.getLogger(f"sr_agent.{__name__}")


@BaseEqGenerator.register("metaai")
class MetaAIEqGenerator(BaseEqGenerator):
    """基于 Meta AI symbolic regression 数据生成方式的方程生成器。

    这是 nd2py 中 MetaAIGenerator 的 scalar-only 简化版：只构造由一元/二元
    运算符和标量变量组成的 unary-binary tree，不处理 nettype。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_operators = self.kwargs.get("n_operators", 5)
        self.operators_to_downsample = self.kwargs.get(
            "operators_to_downsample",
            "Div:0,Arcsin:0,Arccos:0,Tan:0.2,Arctan:0.2,Sqrt:5,Pow2:3,Inv:3",
        )
        self.binary_prob = self._build_operand_prob(self.binary)
        self.unary_prob = self._build_operand_prob(self.unary)

    def generate_eq(self, n_var: int = None, n_operators: int = None) -> nd.Symbol:
        variables = self._get_variables(n_var)
        n_operators = self.n_operators if n_operators is None else n_operators
        if n_operators <= 0:
            return self.generate_leaf(variables, 0)[0]

        sentinel = nd.Identity()
        empty_nodes = list(sentinel.operands)
        next_empty_idx = -1
        n_empty = 1

        while n_operators > 0:
            next_pos, arity = self.generate_next_pos(n_empty, n_operators)
            op = self.generate_ops(arity)
            node = op()

            next_empty_idx += next_pos + 1
            n_empty -= next_pos + 1
            sentinel.replace(empty_nodes[next_empty_idx], node)
            empty_nodes[next_empty_idx] = node
            empty_nodes.extend(node.operands)
            n_empty += op.n_operands
            n_operators -= 1

        n_used_var = 0
        for empty_node in empty_nodes:
            if isinstance(empty_node, nd.Empty):
                leaf, n_used_var = self.generate_leaf(variables, n_used_var)
                sentinel.replace(empty_node, leaf)

        return sentinel.operands[0].copy() # copy to detach from sentinel

    def dist(self, n_op: int, n_empty: int) -> int:
        """Count possible unary-binary trees with n_op operators and n_empty holes."""

        if not hasattr(self, "dp_cache"):
            self.dp_cache = [[0]]
        p1 = 1 if self.unary else 0
        if len(self.dp_cache) <= n_op + n_empty:
            for _ in range(len(self.dp_cache), n_op + n_empty + 1):
                self.dp_cache[0].append(1)
                for r, row in enumerate(self.dp_cache[1:], 1):
                    row.append(row[-1] + p1 * self.dp_cache[r - 1][-2] + self.dp_cache[r - 1][-1])
                self.dp_cache.append([0])
        return self.dp_cache[n_op][n_empty]

    def generate_leaf(
        self,
        variables: Sequence[nd.Variable],
        n_used_var: int,
    ) -> tuple[nd.Variable, int]:
        if n_used_var < len(variables):
            return nd.Variable(variables[n_used_var].name), n_used_var + 1

        idx = int(self._rng.integers(0, len(variables)))
        return nd.Variable(variables[idx].name), n_used_var

    def generate_ops(self, n_operands: int) -> type[nd.Symbol]:
        if n_operands == 1:
            if not self.unary:
                raise ValueError("No unary operands configured.")
            return self._rng.choice(self.unary, p=self.unary_prob)
        elif n_operands == 2:
            if not self.binary:
                raise ValueError("No binary operands configured.")
            return self._rng.choice(self.binary, p=self.binary_prob)
        else:
            raise ValueError(f"Unsupported number of operands: {n_operands}")

    def generate_next_pos(self, n_empty: int, n_operators: int) -> tuple[int, int]:
        if n_empty <= 0:
            raise ValueError("n_empty must be positive.")
        if n_operators <= 0:
            raise ValueError("n_operators must be positive.")

        probs = [self.dist(n_operators - 1, n_empty - i + 1) for i in range(n_empty)]
        if self.unary:
            probs += [self.dist(n_operators - 1, n_empty - i) for i in range(n_empty)]
        probs = np.asarray(probs, dtype=np.float64) / self.dist(n_operators, n_empty)
        next_pos = int(self._rng.choice(len(probs), p=probs))
        n_operands = 1 if next_pos >= n_empty else 2
        next_pos %= n_empty
        return next_pos, n_operands

    def _get_variables(self, n_var: int = None) -> list[nd.Variable]:
        n_var = len(self.variables) if n_var is None else n_var
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        if n_var > len(self.variables):
            raise ValueError(f"n_var={n_var} exceeds configured variables ({len(self.variables)}).")
        return self.variables[:n_var]

    def _build_operand_prob(self, operands: Sequence[type[nd.Symbol]]) -> np.ndarray:
        if not operands:
            return np.asarray([], dtype=np.float64)
        prob_dict = defaultdict(lambda: 1.0)
        for item in self.operators_to_downsample.split(","):
            if item:
                op_name, prob = item.split(":")
                op = getattr(nd, op_name)
                prob_dict[op] = float(prob)
        prob = np.asarray([prob_dict[op] for op in operands], dtype=np.float64)
        if prob.sum() <= 0:
            return np.ones(len(operands), dtype=np.float64) / len(operands)
        return prob / prob.sum()
