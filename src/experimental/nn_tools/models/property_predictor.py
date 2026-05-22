# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from .foundation_model import FoundationModel

__all__ = ["PropertyPredictor"]
VALID_SCOPES = ["target", "variable", "variable_pair"]
VALID_TASKS = ["binary", "ternary", "quaternary", "regression"]


@dataclass(frozen=True)
class PropertySpec:
    """Normalized description of one property prediction target."""
    name: str
    scope: str
    task: str


class PropertyPredictor(FoundationModel):
    """
    Args:
        args.d_model:            Dimension of both input embeddings.
        args.vocab_size:         Number of formula symbols to predict.
        args.nhead:              Number of attention heads.
        args.num_encoder_layers: Number of Transformer encoder layers for data embeddings.
        args.num_decoder_layers: Number of Transformer decoder layers for partial formula embeddings.
        args.dim_feedforward:    Hidden dimension of Transformer feed-forward layers.
        args.dropout:            Dropout probability used by Transformer and positional encodings.
        args.max_formula_len:    Maximum number of formula tokens.
        args.output_pooling:     How to pool decoder outputs before projecting to logits.
                                 Must be one of ``"attention"``, ``"average"`` or ``"last"``.
        args.mlp_hidden_dims:    Hidden dimensions for the MLP layers in the decoder.
        args.encoder_pooling:    How to pool encoder outputs into a single vector. 
                                 Must be one of ``"attention"``, ``"average"`` or ``"last"``.
        args.shared_decoder_mlp  Whether to use a single shared MLP for all properties (True) or separate MLPs (False).
    """
    def __init__(self, args, properties) -> None:
        super().__init__(args)
        if not properties:
            raise ValueError("At least one property spec is required.")
        if args.encoder_pooling not in (_VALID := ["attention", "average", "last"]):
            raise ValueError(f"encoder_pooling must be one of {_VALID!r}, got {args.encoder_pooling!r}.")
        if any(dim <= 0 for dim in args.mlp_hidden_dims):
            raise ValueError("All mlp_hidden_dims must be positive.")

        self.properties = self._normalize_properties(properties)
        self.encoder_pooling = args.encoder_pooling
        self.shared_decoder_mlp = args.shared_decoder_mlp
        if self.encoder_pooling == "attention":
            self.encoder_attention_score = nn.Linear(self.d_model, 1)

        task_to_output_dim = {'binary': 2, 'ternary': 3, 'quaternary': 4, "regression": 1}
        scope_to_output_dim = {'target': 1, 'variable': args.max_var_num, 'variable_pair': args.max_var_num ** 2}
        self.output_dims = {prop.name: scope_to_output_dim[prop.scope] * task_to_output_dim[prop.task] for prop in self.properties}
        if self.shared_decoder_mlp:
            out_dim = sum(self.output_dims.values())
            self.decoder_mlp = self._build_mlp(self.d_model, args.mlp_hidden_dims, out_dim)
        else:
            self.decoder_mlps = nn.ModuleDict({
                spec.name: self._build_mlp(self.d_model, args.mlp_hidden_dims, self.output_dims[spec.name])
                for spec in self.properties
            })

    def forward(
        self,
        data_embedding: torch.Tensor,
        data_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict all configured properties.

        Args:
            data_embedding: Embedded data tokens with shape
                ``(batch, data_len, d_model)``.
            data_padding_mask: Optional bool mask with shape
                ``(batch, data_len)``. ``True`` marks padded data tokens.
        Returns:
            Mapping from property name to model output. Each output has shape
            ``(batch, scope_size * task_size)``, where ``scope_size`` is ``1``
            for ``target``, ``args.max_var_num`` for ``variable`` and
            ``args.max_var_num ** 2`` for ``variable_pair``. ``task_size`` is
            ``2``/``3``/``4`` for classification tasks and ``1`` for regression.
        """
        memory = self.encoder(
            data_embedding,
            src_key_padding_mask=data_padding_mask,
        )

        if self.encoder_pooling == "last":
            if data_padding_mask is None:
                pooled =  memory[:, -1, :]
            else:
                lengths = (~data_padding_mask).sum(dim=1).clamp_min(1)
                batch_idx = torch.arange(memory.shape[0], device=memory.device)
                pooled =  memory[batch_idx, lengths.to(memory.device) - 1, :]
        elif self.encoder_pooling == "average":
            if data_padding_mask is None:
                pooled =  memory.mean(dim=1)
            else:
                valid_mask = ~data_padding_mask
                lengths = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled =  (memory * valid_mask.unsqueeze(-1)).sum(dim=1) / lengths.to(memory.dtype)
        elif self.encoder_pooling == "attention":
            score = self.encoder_attention_score(memory).squeeze(-1)
            if data_padding_mask is not None:
                score = score.masked_fill(data_padding_mask, float("-inf"))
                all_masked = data_padding_mask.all(dim=1)
                score = score.masked_fill(all_masked.unsqueeze(1), 0.0)
            weight = torch.softmax(score, dim=1).unsqueeze(-1)
            pooled =  (memory * weight).sum(dim=1)
        else:
            raise ValueError(f"Invalid encoder_pooling: {self.encoder_pooling!r}")

        if self.shared_decoder_mlp:
            features = self.decoder_mlp(pooled)
            splited = torch.split(features, [self.output_dims[spec.name] for spec in self.properties], dim=-1)
            return {spec.name: split for spec, split in zip(self.properties, splited)}
        else:
            return {spec.name: self.decoder_mlps[spec.name](pooled) for spec in self.properties}

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _normalize_properties(properties: Iterable[Mapping[str, Any]]) -> list[PropertySpec]:
        normalized = {}
        for row in properties:
            if (name := row["name"]) in normalized:
                raise ValueError(f"Duplicate property name: {name!r}.")
            if (scope := row["scope"]) not in VALID_SCOPES:
                raise ValueError(f"Invalid scope for {name!r}: {scope!r}. Use one of {VALID_SCOPES!r}.")
            if (task := row.get("task")) not in VALID_TASKS:
                raise ValueError(f"Invalid task for {name!r}: {task!r}. Use one of {VALID_TASKS!r}.")
            normalized[name] = PropertySpec(name=name, scope=scope, task=task)
        return list(normalized.values())
