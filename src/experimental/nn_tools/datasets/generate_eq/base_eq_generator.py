# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import logging
import numpy as np
import nd2py as nd
from abc import abstractmethod
from typing import List, Sequence, Type
from sr_agent.utils import FactoryMixin

_logger = logging.getLogger(f'sr_agent.{__name__}')


class BaseEqGenerator(FactoryMixin):
    def __init__(
        self,
        n_variables: int | None = None,
        variables: Sequence[str | nd.Variable] = None,
        binary: Sequence[Type[nd.Symbol]] = [nd.Add, nd.Sub, nd.Mul, nd.Div],
        unary: Sequence[Type[nd.Symbol]] = [nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
        random_seed=None,
        **kwargs,
    ):
        if (n_variables is not None) and (variables is not None):
            _logger.warning("Both n_variables and variables are provided. n_variables will be ignored.")
            n_variables = len(variables)
        elif (n_variables is not None) and (variables is None):
            variables = [f"x{i}" for i in range(n_variables)]
        elif (n_variables is None) and (variables is not None):
            n_variables = len(variables)
        elif (n_variables is None) and (variables is None):
            n_variables = 3
            variables = [f"x{i}" for i in range(n_variables)]

        self.kwargs = kwargs
        self.random_seed = random_seed if random_seed is not None else np.random.randint(1e9)
        self._rng = np.random.default_rng(self.random_seed)
        self.variables = self._normalize_variables(variables)
        self.binary = self._normalize_operands(binary)
        self.unary = self._normalize_operands(unary)
        self.symbols = self.binary + self.unary

    def __call__(self, _rng=None) -> nd.Symbol:
        """ 生成一个随机方程。"""
        if _rng is not None:
            _rng_old, self._rng = self._rng, _rng
        try:
            return self.generate_eq()
        finally:
            if _rng is not None:
                self._rng = _rng_old

    @abstractmethod
    def generate_eq(self) -> nd.Symbol:
        raise NotImplementedError("Subclasses must implement generate_eq method")

    @staticmethod
    def _normalize_variables(variables: Sequence[str | nd.Variable]) -> List[nd.Variable]:
        return [var if isinstance(var, nd.Variable) else nd.Variable(var) for var in variables]
    
    @staticmethod
    def _normalize_operands(operands: Sequence[Type[nd.Symbol]]) -> List[nd.Symbol]:
        """ nd.Add -> nd.Add, "Add" -> nd.Add, "add" -> nd.Add """
        normalized = []
        for op in operands:
            if isinstance(op, type) and issubclass(op, nd.Symbol):
                normalized.append(op)
            elif isinstance(op, nd.Symbol) and op.n_operands > 0:
                normalized.append(type(op))
            elif isinstance(op, nd.Symbol) and op.n_operands == 0:
                raise ValueError(f"Operand {op} is a leaf symbol, which is not allowed in binary/unary operands.")
            elif not isinstance(op, str):
                raise ValueError(f"Invalid operand: {op}")
            elif (op_class := getattr(nd, op, None)) is not None and issubclass(op_class, nd.Symbol):
                normalized.append(op_class)
            else:
                raise ValueError(f"Unknown symbol: {op}")
        return normalized
