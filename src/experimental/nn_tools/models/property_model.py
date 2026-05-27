# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Property prediction model v3: data -> property labels (4-class encoding).

v3 architecture improvements over v2:
  1. Per-variable cross-attention: each variable gets its own representation
     via learned query vectors attending to encoder memory, instead of all
     variables sharing one pooled vector.
  2. Deeper classification heads with LayerNorm (2 hidden layers).
  3. Higher default dropout (0.2) for better regularization.
  4. Label smoothing support in loss computation.

Label encoding:
  Monotonicity: 4 classes (0=non-mono/unknown, 1=inc, 2=dec, 3=const)
  Convexity:    4 classes (0=neither/unknown, 1=convex, 2=concave, 3=affine)
  Periodicity:  2 classes (0=non-periodic, 1=periodic)
  Mul-sep:      2 classes (0=not separable, 1=separable)
"""
import torch
import torch.nn as nn


class PropertyPredictionModel(nn.Module):
    """Predict formula properties from data embeddings (v3 architecture)."""

    N_MONO_CLASSES = 4
    N_CONV_CLASSES = 4
    N_PERIOD_CLASSES = 2
    N_SEP_CLASSES = 2

    def __init__(self, args) -> None:
        super().__init__()
        d = args.d_model
        self.d_model = d
        self.max_var_num = args.max_var_num
        dropout = getattr(args, "dropout", 0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_encoder_layers,
        )

        # Global attention pool (for formula-level sep head)
        self.global_pool_attn = nn.Linear(d, 1)
        self.global_pool_norm = nn.LayerNorm(d)

        # Per-variable cross-attention: learned queries for each variable slot
        self.var_queries = nn.Parameter(torch.randn(args.max_var_num, d) * 0.02)
        self.var_cross_attn = nn.MultiheadAttention(
            d, args.nhead, dropout=dropout, batch_first=True,
        )
        self.var_norm = nn.LayerNorm(d)

        # Per-variable heads: input is per-variable representation (B, max_var, d)
        self.head_mono = self._build_head(d, self.N_MONO_CLASSES, dropout)
        self.head_conv = self._build_head(d, self.N_CONV_CLASSES, dropout)
        self.head_period = self._build_head(d, self.N_PERIOD_CLASSES, dropout)

        # Formula-level head: input is global pooled (B, d)
        self.head_sep = self._build_head(d, self.N_SEP_CLASSES, dropout)

    @staticmethod
    def _build_head(d: int, n_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_classes),
        )

    def forward(
        self,
        data_embedding: torch.Tensor,
        data_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encoder(data_embedding, src_key_padding_mask=data_padding_mask)

        # Global pool for formula-level prediction
        score = self.global_pool_attn(memory).squeeze(-1)
        if data_padding_mask is not None:
            score = score.masked_fill(data_padding_mask, float("-inf"))
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = self.global_pool_norm((memory * weight).sum(dim=1))  # (B, d)

        # Per-variable cross-attention
        B = memory.shape[0]
        var_q = self.var_queries.unsqueeze(0).expand(B, -1, -1)  # (B, max_var, d)
        var_repr, _ = self.var_cross_attn(var_q, memory, memory,
                                          key_padding_mask=data_padding_mask)
        var_repr = self.var_norm(var_repr)  # (B, max_var, d)

        # Per-variable classification
        mono = self.head_mono(var_repr)   # (B, max_var, 4)
        conv = self.head_conv(var_repr)   # (B, max_var, 4)
        period = self.head_period(var_repr)  # (B, max_var, 2)

        # Formula-level classification
        sep = self.head_sep(pooled)  # (B, 2)

        return {
            "monotonicity": mono,
            "convexity": conv,
            "periodicity": period,
            "multiplicative_separable": sep,
        }

    def load_encoder_from_foundation(self, checkpoint_path: str, device="cpu"):
        """Load encoder weights from a FoundationModel checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        encoder_state = {}
        model_state = ckpt.get("model", ckpt)
        for k, v in model_state.items():
            if k.startswith("encoder."):
                encoder_state[k] = v
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        return missing, unexpected
