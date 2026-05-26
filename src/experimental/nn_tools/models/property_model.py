# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Property prediction model: data -> property labels (4-class encoding).

Reuses the same encoder architecture as FoundationModel but replaces the
formula decoder with per-property classification heads.

Label encoding:
  Monotonicity: 4 classes (0=non-mono/unknown, 1=inc, 2=dec, 3=const)
  Convexity:    4 classes (0=neither/unknown, 1=convex, 2=concave, 3=affine)
  Periodicity:  2 classes (0=non-periodic, 1=periodic)
  Mul-sep:      2 classes (0=not separable, 1=separable)
"""
import torch
import torch.nn as nn


class PropertyPredictionModel(nn.Module):
    """Predict formula properties from data embeddings."""

    N_MONO_CLASSES = 4
    N_CONV_CLASSES = 4
    N_PERIOD_CLASSES = 2
    N_SEP_CLASSES = 2

    def __init__(self, args) -> None:
        super().__init__()
        d = args.d_model
        self.d_model = d
        self.max_var_num = args.max_var_num

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_encoder_layers,
        )

        self.pool_attention = nn.Linear(d, 1)

        head_hidden = d
        self.head_mono = nn.Sequential(
            nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(head_hidden, self.max_var_num * self.N_MONO_CLASSES),
        )
        self.head_conv = nn.Sequential(
            nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(head_hidden, self.max_var_num * self.N_CONV_CLASSES),
        )
        self.head_period = nn.Sequential(
            nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(head_hidden, self.max_var_num * self.N_PERIOD_CLASSES),
        )
        self.head_sep = nn.Sequential(
            nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(head_hidden, self.N_SEP_CLASSES),
        )

    def forward(
        self,
        data_embedding: torch.Tensor,
        data_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encoder(data_embedding, src_key_padding_mask=data_padding_mask)

        score = self.pool_attention(memory).squeeze(-1)
        if data_padding_mask is not None:
            score = score.masked_fill(data_padding_mask, float("-inf"))
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (memory * weight).sum(dim=1)  # (B, d)

        B = pooled.shape[0]
        mono = self.head_mono(pooled).view(B, self.max_var_num, self.N_MONO_CLASSES)
        conv = self.head_conv(pooled).view(B, self.max_var_num, self.N_CONV_CLASSES)
        period = self.head_period(pooled).view(B, self.max_var_num, self.N_PERIOD_CLASSES)
        sep = self.head_sep(pooled)

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
