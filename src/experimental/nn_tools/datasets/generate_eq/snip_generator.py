# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import logging

import nd2py as nd

from .base_eq_generator import BaseEqGenerator
from .metaai_generator import MetaAIEqGenerator

_logger = logging.getLogger(f"sr_agent.{__name__}")


@BaseEqGenerator.register("snip")
class SNIPEqGenerator(MetaAIEqGenerator):
    """SNIP 风格方程生成器。

    先生成一个二元运算树，再添加一元运算和线性 prefactor。这里保留 nd2py
    版本的核心思路，但只处理 scalar 符号。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_var = self.kwargs.get("max_var", 5)
        self.min_unary = self.kwargs.get("min_unary", 0)
        self.max_unary = self.kwargs.get("max_unary", 4)
        self.min_binary_per_var = self.kwargs.get("min_binary_per_var", 0)
        self.max_binary_per_var = self.kwargs.get("max_binary_per_var", 1)
        self.max_binary_ops_offset = self.kwargs.get("max_binary_ops_offset", 4)
        self.max_unary_depth = self.kwargs.get("max_unary_depth", 6)
        self.n_mantissa = self.kwargs.get("n_mantissa", 4)
        self.max_exp = self.kwargs.get("max_exp", 1)
        self.min_exp = self.kwargs.get("min_exp", 0)

    def generate_eq(
        self,
        n_var: int = None,
        n_unary: int = None,
        n_binary: int = None,
    ) -> nd.Symbol:
        n_var = self._sample_n_var(n_var)
        n_unary = self._sample_n_unary(n_unary)
        n_binary = self._sample_n_binary(n_binary, n_var)

        unary_backup = self.unary
        unary_prob_backup = self.unary_prob
        self.unary = []
        self.unary_prob = []
        eqtree = super().generate_eq(n_var=n_var, n_operators=n_binary)
        self.unary = unary_backup
        self.unary_prob = unary_prob_backup

        eqtree = self.add_unaries(eqtree, n_unary)
        eqtree = self.add_prefactors(eqtree)
        return eqtree

    def generate_float(self) -> nd.Number:
        sign = self._rng.choice([-1, 1])
        mantissa = self._rng.integers(1, 10**self.n_mantissa) / 10 ** (self.n_mantissa - 1)
        exponent = self._rng.integers(self.min_exp, self.max_exp + 1)
        return nd.Number(sign * mantissa * 10**exponent)

    def add_unaries(self, eqtree: nd.Symbol, n_unary: int) -> nd.Symbol:
        if n_unary <= 0 or not self.unary:
            return eqtree

        for _ in range(n_unary):
            candidates = [
                symbol
                for symbol in eqtree.iter_preorder()
                if not isinstance(symbol, nd.Empty) and len(symbol) < self.max_unary_depth
            ]
            if not candidates:
                break
            target = self._choice(candidates)
            unary = self._rng.choice(self.unary, p=self.unary_prob)
            wrapped = unary(target.copy())
            if target is eqtree:
                eqtree = wrapped
            else:
                eqtree.replace(target, wrapped)
        return eqtree

    def add_prefactors(self, eqtree: nd.Symbol) -> nd.Symbol:
        return nd.Add(self.generate_float(), self._add_prefactors(eqtree))

    def _add_prefactors(self, eqtree: nd.Symbol) -> nd.Symbol:
        if isinstance(eqtree, (nd.Variable, nd.Number)):
            return eqtree.copy()

        if isinstance(eqtree, (nd.Add, nd.Sub)):
            x1, x2 = eqtree.operands
            x1 = self._add_prefactors(x1)
            x2 = self._add_prefactors(x2)
            if not isinstance(x1, (nd.Add, nd.Sub)):
                x1 = nd.Mul(self.generate_float(), x1)
            if not isinstance(x2, (nd.Add, nd.Sub)):
                x2 = nd.Mul(self.generate_float(), x2)
            return type(eqtree)(x1, x2)

        if type(eqtree) in self.unary:
            x = self._add_prefactors(eqtree.operands[0])
            if not isinstance(x, (nd.Add, nd.Sub)):
                x = nd.Add(self.generate_float(), nd.Mul(self.generate_float(), x))
            return type(eqtree)(x)

        operands = [self._add_prefactors(operand) for operand in eqtree.operands]
        return type(eqtree)(*operands)

    def _sample_n_var(self, n_var: int = None) -> int:
        if n_var is not None:
            return n_var
        high = min(self.max_var, len(self.variables))
        if high <= 0:
            raise ValueError("No variables configured.")
        return int(self._rng.integers(1, high + 1))

    def _sample_n_unary(self, n_unary: int = None) -> int:
        if n_unary is not None:
            return n_unary
        return int(self._rng.integers(self.min_unary, self.max_unary + 1))

    def _sample_n_binary(self, n_binary: int = None, n_var: int = None) -> int:
        if n_binary is not None:
            return n_binary
        low = self.min_binary_per_var * n_var
        high = self.max_binary_per_var * n_var + self.max_binary_ops_offset
        high = max(high, low + 1)
        return int(self._rng.integers(low, high))

    def _choice(self, values):
        if not values:
            raise ValueError("Cannot choose from an empty sequence.")
        return values[int(self._rng.integers(0, len(values)))]


@BaseEqGenerator.register("snip2")
class SNIPEqGenerator2(SNIPEqGenerator):
    def add_prefactors(self, eqtree: nd.Symbol) -> nd.Symbol:
        a = self.generate_float()
        b = self.generate_float()
        return nd.Add(nd.Mul(a, self._affine_variables(eqtree)), b)

    def _affine_variables(self, eqtree: nd.Symbol) -> nd.Symbol:
        if isinstance(eqtree, nd.Variable):
            a = self.generate_float()
            b = self.generate_float()
            return nd.Add(nd.Mul(a, eqtree.copy()), b)
        if isinstance(eqtree, nd.Number):
            return eqtree.copy()
        return type(eqtree)(*[self._affine_variables(operand) for operand in eqtree.operands])


@BaseEqGenerator.register("snip3")
class SNIPEqGenerator3(SNIPEqGenerator):
    def generate_eq(
        self,
        n_var: int = None,
        n_unary: int = None,
        n_binary: int = None,
    ) -> nd.Symbol:
        n_var = self._sample_n_var(n_var)
        n_unary = self._sample_n_unary(n_unary)
        n_binary = self._sample_n_binary(n_binary, n_var)

        unary_backup = self.unary
        unary_prob_backup = self.unary_prob
        self.unary = []
        self.unary_prob = []
        eqtree = super(SNIPEqGenerator, self).generate_eq(n_var=n_var, n_operators=n_binary)
        self.unary = unary_backup
        self.unary_prob = unary_prob_backup

        return self.add_unaries(eqtree, n_unary)

    def generate_float(self) -> nd.Number:
        raise NotImplementedError("SNIPEqGenerator3 does not generate numeric prefactors.")
