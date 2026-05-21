# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import logging
import numpy as np
import nd2py as nd
from abc import abstractmethod
from typing import List, Sequence, Type
from sr_agent.utils import FactoryMixin
from .base_eq_generator import BaseEqGenerator

_logger = logging.getLogger(f'sr_agent.{__name__}')


@BaseEqGenerator.register("gplearn")
class GplearnEqGenerator(BaseEqGenerator):
    """基于 gplearn 的方程生成器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full_prob = self.kwargs.get("full_prob", 0.5)
        self.depth_range = self.kwargs.get("depth_range", (2, 6))
        self.const_range = self.kwargs.get("const_range", None)
        self.n_var_range = self.kwargs.get("n_var_range", (1, len(self.variables) + 1))

    def generate_eq(self) -> nd.Symbol:
        n_var = self._rng.integers(*self.n_var_range)
        variables = self._get_variables(n_var)
        full_tree = self._rng.random() < self.full_prob
        max_depth = self._rng.integers(*self.depth_range)
        op_prob = 1.0 if full_tree else len(self.symbols) / (len(variables) + len(self.symbols))

        # Start a eqtree with a function to avoid degenerative eqtrees
        eqtree = self.generate_node()
        empty_nodes_and_depth = [(i, 1) for i in eqtree.operands]

        while empty_nodes_and_depth:
            empty_node, depth = empty_nodes_and_depth.pop(0)
            if (depth < max_depth) and (self._rng.random() < op_prob):
                node = self.generate_node()
                eqtree.replace(empty_node, node)
                empty_nodes_and_depth.extend([(i, depth + 1) for i in node.operands])
            else:  # Variable or Number
                leaf = self.generate_leaf(variables)
                eqtree.replace(empty_node, leaf)
        self._ensure_has_variable(eqtree, variables)
        return eqtree

    def generate_node(self) -> nd.Symbol:
        symbol = self._choice(self.symbols)
        return symbol()

    def generate_leaf(self, variables: Sequence[nd.Variable]) -> nd.Number | nd.Variable:
        leafs = list(variables)
        if self.const_range is not None:
            const_range = self.const_range
        elif not leafs:
            const_range = (-1, 1)
        else:
            const_range = None
        if const_range is not None:
            leafs.append(nd.Number(self._rng.uniform(*const_range)))
        return self._choice(leafs)

    def _get_variables(self, n_var: int = None) -> List[nd.Variable]:
        if n_var is None:
            n_var = len(self.variables)
        if self.variables is not None:
            if n_var is not None and n_var > len(self.variables):
                raise ValueError(f"n_var={n_var} exceeds configured variables ({len(self.variables)}).")
            return self.variables[:n_var] if n_var is not None else list(self.variables)
        if n_var is None:
            raise ValueError("n_var must be provided when variables are not configured.")
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        return [nd.Variable(f"x{i}") for i in range(n_var)]

    def _choice(self, values: Sequence):
        if not values:
            raise ValueError("Cannot choose from an empty sequence.")
        return values[int(self._rng.integers(0, len(values)))]

    def _ensure_has_variable(self, eqtree: nd.Symbol, variables: Sequence[nd.Variable]) -> None:
        if any(isinstance(symbol, nd.Variable) for symbol in eqtree.iter_preorder()):
            return
        if not variables:
            return
        for symbol in eqtree.iter_postorder():
            if isinstance(symbol, nd.Number):
                eqtree.replace(symbol, self._choice(variables))
                return
