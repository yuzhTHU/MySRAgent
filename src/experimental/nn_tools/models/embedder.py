from __future__ import annotations
from itertools import count
import torch
import logging
import numpy as np
import nd2py as nd
import torch.nn as nn
from typing import List, Mapping, Sequence, Type

_logger = logging.getLogger(f"sr_agent.{__name__}")

__all__ = [
    "DataEmbedder",
    "EquationEmbedder",
    "FloatEmbedder",
]


class FloatEmbedder(nn.Module):
    """将浮点数编码为嵌入，避免直接使用浮点数值，后者分布范围太大，且包括 Nan / Inf 等特殊值，可能会导致训练不稳定。

    A finite float is decomposed into three discrete parts:
    - sign: ``+`` or ``-``
    - mantissa: ``N0000`` ... ``N9999`` when ``mantissa_digits=4``
    - exponent: ``E-99`` ... ``E+99`` when ``min_exponent=-99`` and
      ``max_exponent=99``

    ``inf``, ``-inf`` and ``nan`` bypass the decomposition and use dedicated
    learnable embeddings.
    """

    SPECIAL_POS_INF = 0
    SPECIAL_NEG_INF = 1
    SPECIAL_NAN = 2

    def __init__(
        self,
        d_model: int,
        *,
        mantissa_digits: int = 4,
        min_exponent: int = -99,
        max_exponent: int = 99,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if mantissa_digits <= 0:
            raise ValueError("mantissa_digits must be positive.")
        if min_exponent > max_exponent:
            raise ValueError("min_exponent must be <= max_exponent.")

        self.d_model = d_model
        self.mantissa_digits = mantissa_digits
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.num_mantissas = 10**mantissa_digits
        self.num_exponents = max_exponent - min_exponent + 1

        self.sign_embedding = nn.Embedding(2, d_model)
        self.mantissa_embedding = nn.Embedding(self.num_mantissas, d_model)
        self.exponent_embedding = nn.Embedding(self.num_exponents, d_model)
        self.special_embedding = nn.Embedding(3, d_model)

    def forward(self, value: float | np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        out = torch.empty(flat.shape[0], self.d_model, device=flat.device)

        pos_inf_mask = torch.isposinf(flat)
        neg_inf_mask = torch.isneginf(flat)
        nan_mask = torch.isnan(flat)
        finite_mask = torch.isfinite(flat)

        if finite_mask.any():
            finite_values = flat[finite_mask]
            sign_idx, mantissa_idx, exponent_idx = self.decompose_finite(finite_values)
            out[finite_mask] = (
                self.sign_embedding(sign_idx)
                + self.mantissa_embedding(mantissa_idx)
                + self.exponent_embedding(exponent_idx)
            )

        if pos_inf_mask.any():
            out[pos_inf_mask] = self.special_embedding(
                torch.full(
                    (int(pos_inf_mask.sum()),),
                    self.SPECIAL_POS_INF,
                    dtype=torch.long,
                    device=flat.device,
                )
            )
        if neg_inf_mask.any():
            out[neg_inf_mask] = self.special_embedding(
                torch.full(
                    (int(neg_inf_mask.sum()),),
                    self.SPECIAL_NEG_INF,
                    dtype=torch.long,
                    device=flat.device,
                )
            )
        if nan_mask.any():
            out[nan_mask] = self.special_embedding(
                torch.full(
                    (int(nan_mask.sum()),),
                    self.SPECIAL_NAN,
                    dtype=torch.long,
                    device=flat.device,
                )
            )

        return out.reshape(*original_shape, self.d_model)

    @property
    def device(self) -> torch.device:
        return self.sign_embedding.weight.device

    def decompose_finite(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sign_idx = (value < 0).long()
        abs_value = value.abs()
        nonzero = abs_value > 0

        exponent = torch.zeros_like(abs_value, dtype=torch.long)
        exponent[nonzero] = torch.floor(torch.log10(abs_value[nonzero])).long()
        exponent = exponent.clamp(self.min_exponent, self.max_exponent)

        scale = torch.pow(torch.tensor(10.0, device=value.device), exponent.float())
        normalized = torch.where(nonzero, abs_value / scale, torch.ones_like(abs_value))
        mantissa_float = ((normalized - 1.0) / 9.0) * (self.num_mantissas - 1)
        mantissa_idx = mantissa_float.round().long().clamp(0, self.num_mantissas - 1)

        exponent_idx = exponent - self.min_exponent
        return sign_idx, mantissa_idx, exponent_idx


class DataEmbedder(nn.Module):
    """将 data_dict 中的数值编码为嵌入，每组 (x_1, x_2, .., x_n, y) 编码为一个嵌入向量。

    ``data_dict`` is first stacked into a 2D array with shape
    ``(array_length, len(variables))``. Every float is encoded by
    ``FloatEmbedder``. Embeddings from the same row are then pooled into one
    embedding, so the return shape is ``(array_length, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        float_embedder: FloatEmbedder | None = None,
        pooling: str = "attention",
        mantissa_digits: int = 4,
        min_exponent: int = -99,
        max_exponent: int = 99,
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "average", "sum"}:
            raise ValueError(f"pooling must be one of: 'attention', 'average', 'sum', got {pooling!r}.")
        self.d_model = d_model
        self.float_embedder = float_embedder or FloatEmbedder(
            d_model,
            mantissa_digits=mantissa_digits,
            min_exponent=min_exponent,
            max_exponent=max_exponent,
        )
        self.pooling = pooling
        if pooling == "attention":
            self.attention_score = nn.Linear(d_model, 1)

    def forward(self, data_dict: Mapping[str, np.ndarray], variables: Sequence[str]) -> torch.Tensor:
        if not variables:
            raise ValueError("variables must not be empty.")
        arrays: List[np.ndarray] = []
        expected_len = None
        for variable in variables:
            if variable not in data_dict:
                raise KeyError(f"Variable {variable!r} not found in data_dict.")
            elif (array := np.asarray(data_dict[variable])).ndim != 1:
                raise ValueError(f"Variable {variable!r} must be a 1D np.ndarray, got shape {array.shape}.")
            elif (expected_len := expected_len or len(array)) != len(array):
                raise ValueError(
                    f"All arrays must have the same length. Variable {variable!r} has "
                    f"length {len(array)}, expected {expected_len}."
                )
            else:
                arrays.append(array.astype(np.float32, copy=False))
        values = np.stack(arrays, axis=-1)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.float_embedder.device)
        value_emb = self.float_embedder(tensor)
        return self.pool(value_emb)

    def pool(self, value_emb: torch.Tensor) -> torch.Tensor:
        if self.pooling == "average":
            return value_emb.mean(dim=1)
        elif self.pooling == "sum":
            return value_emb.sum(dim=1)
        elif self.pooling == "attention":
            score = self.attention_score(value_emb).squeeze(-1)
            weight = torch.softmax(score, dim=1).unsqueeze(-1)
            return (value_emb * weight).sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling!r}.")


class EquationEmbedder(nn.Module):
    """将符号公式（nd2py.Symbol）中的每个元素编码为一个嵌入向量。

    ``tokenize`` converts an nd2py equation to string tokens, 
    ``vectorize`` converts tokens to ids, and 
    ``forward`` embeds ids with one shared embedding table.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    NUM_TOKEN = "<NUM>"

    def __init__(
        self,
        d_model: int,
        *,
        traversal: str = "postorder",
        operands: Sequence[Type[nd.Symbol]] = (nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Sin, nd.Cos, nd.Exp, nd.Log),
        max_variables: int = 10,
    ) -> None:
        super().__init__()
        if traversal not in {"postorder", "preorder"}:
            raise ValueError("traversal must be 'postorder' or 'preorder'.")
        if max_variables <= 0:
            raise ValueError("max_variables must be positive.")

        self.d_model = d_model
        self.traversal = traversal
        self.max_variables = max_variables
        self.operands = self._normalize_operands(operands)
        self.token2index: dict[str, int] = {}
        self.index2token: dict[int, str] = {}
        
        token_id_iter = count(0)
        def register_token(token: str, index: int | None = None) -> int:
            """ 将 token 注册到词表中，并返回对应的 id。如果 token 已经注册过了，则直接返回已注册的 id。 """
            if token in self.token2index:
                return self.token2index[token]
            token_id = next(token_id_iter) if index is None else index
            self.token2index[token] = token_id
            self.index2token[token_id] = token
            return token_id

        self.pad_token_id = register_token(self.PAD_TOKEN)
        self.unk_token_id = register_token(self.UNK_TOKEN)
        self.variable_ids = [next(token_id_iter) for _ in range(self.max_variables)]
        self._next_variable_slot = 0
        self.operand_to_id = {
            operand: register_token(operand.__name__) for operand in self.operands
        }
        num_symbol_embeddings = next(token_id_iter)
        self.symbol_embedding = nn.Embedding(
            num_symbol_embeddings, d_model, padding_idx=self.pad_token_id,
        )

    def tokenize(self, eq: nd.Symbol) -> list[str | float]:
        """Convert an nd2py equation into a flat token sequence.
        Variables are emitted as their names, 
        numeric constants as floats, and
        operators as nd2py class names. 
        Token order follows ``self.traversal``.
        """
        if self.traversal == "postorder":
            flatten = [symbol for symbol in eq.iter_postorder()]
        elif self.traversal == "preorder":
            flatten = [symbol for symbol in eq.iter_preorder()]
        else:
            raise ValueError(f"traversal must be 'postorder' or 'preorder', got {self.traversal!r}.")

        tokens = []
        for symbol in flatten:
            if isinstance(symbol, nd.Variable):
                tokens.append(symbol.name)
            elif isinstance(symbol, nd.Number):
                tokens.append(float(symbol.value))
            else:
                tokens.append(type(symbol).__name__)
        return tokens

    def vectorize(self, tokens: Sequence[str | float], variables: List[str] = None) -> list[int | float]:
        """Convert equation tokens to vocabulary ids."""
        if len(variables) > self.max_variables:
            raise ValueError(f"Number of variables {len(variables)} exceeds max_variables {self.max_variables}.")
        ids = []
        for token in tokens:
            if not isinstance(token, (str, float)):
                raise TypeError(f"Equation tokens must be strings or floats, got {type(token).__name__}.")
            elif token in self.token2index:
                ids.append(self.token2index[token])
            elif isinstance(token, float):
                ids.append(token)
            elif token in variables:
                ids.append(self.variable_ids[variables.index(token)])
            else:
                _logger.warning(f"Token {token!r} not found in token2index and variables. Using UNK token id.")
                ids.append(self.unk_token_id)
        return ids

    def forward(
        self,
        eq:    nd.Symbol     | Sequence[nd.Symbol]                    = None,
        token: Sequence[str] | Sequence[Sequence[str]] | torch.Tensor = None,
        index: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor = None,
        variables: List[str] = None,
    ) -> torch.Tensor:
        """Embed equations, token sequences, or index sequences.

        Exactly one of ``eq``, ``token`` or ``index`` must be provided.
        If ``eq`` is provided, it is tokenized and vectorized before embedding.
        If ``token`` is provided, it is vectorized before embedding.
        If ``index`` is provided, it is directly embedded.

        Returns:
            ``embedding`` has shape ``(batch_size, max_num_tokens, d_model)`` and is zero-padded.
        """
        if sum(x is not None for x in (eq, token, index)) != 1:
            raise ValueError("Exactly one of eq, token or index must be provided.")
        if ((eq is not None) or (token is not None)) and (variables is None):
            raise ValueError("variables must be provided when eq or token is given.")
        
        # 获取 index_list: list[list[int | float]]
        if eq is not None:
            eq_list = eq if (is_list := isinstance(eq, list)) else [eq]
            token_list = [self.tokenize(eq) for eq in eq_list]
            index_list = [self.vectorize(tokens, variables) for tokens in token_list]
        elif token is not None:
            token_list = token if (is_list := isinstance(token[0], list)) else [token]
            index_list = [self.vectorize(tokens, variables) for tokens in token_list]
        elif index is not None:
            index_list = index if (is_list := isinstance(index[0], list)) else [index]
        
        # 合并成一个 batch
        foo = {'dtype': torch.long, 'device': self.symbol_embedding.weight.device}
        if is_list:
            lengths = torch.tensor([len(index) for index in index_list], **foo)
            max_len = int(lengths.max().item())
            batch_size = len(index_list)
            index_tensor = torch.full((batch_size, max_len), self.pad_token_id, **foo)
            for batch_idx, index in enumerate(index_list):
                index_tensor[batch_idx, :len(index)] = torch.tensor(index, **foo)
        else:
            index_tensor = torch.tensor(index_list[0], **foo)

        # 编码
        output = self.symbol_embedding(index_tensor)
        return output

    @staticmethod
    def _normalize_operands(operands: Sequence[Type[nd.Symbol]]) -> List[Type[nd.Symbol]]:
        """ nd.Add -> nd.Add, "Add" -> nd.Add, "add" -> nd.Add """
        normalized = []
        for op in operands:
            if isinstance(op, type) and issubclass(op, nd.Symbol):
                normalized.append(op)
            elif isinstance(op, nd.Symbol) and op.n_operands > 0:
                normalized.append(type(op))
            elif isinstance(op, nd.Symbol) and op.n_operands == 0:
                raise ValueError(
                    f"Operand {op} is a leaf symbol, which is not allowed in binary/unary operands."
                )
            elif not isinstance(op, str):
                raise ValueError(f"Invalid operand: {op}")
            elif (
                (op_class := getattr(nd, op, None)) is not None
                and issubclass(op_class, nd.Symbol)
            ):
                normalized.append(op_class)
            else:
                raise ValueError(f"Unknown symbol: {op}")
        return normalized
