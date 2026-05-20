# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import torch
import torch.nn as nn
from .pe import PositionalEncoding


class FoundationModel(nn.Module):
    """Transformer model for next-symbol prediction from data/formula embeddings.

    Args:
        d_model: Dimension of both input embeddings.
        vocab_size: Number of formula symbols to predict.
        nhead: Number of attention heads.
        num_encoder_layers: Number of Transformer encoder layers for data embeddings.
        num_decoder_layers: Number of Transformer decoder layers for partial formula embeddings.
        dim_feedforward: Hidden dimension of Transformer feed-forward layers.
        dropout: Dropout probability used by Transformer and positional encodings.
        max_formula_len: Maximum number of formula tokens.
        output_pooling: How to pool decoder outputs before projecting to logits.
          Must be one of ``"attention"``, ``"average"`` or ``"last"``.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_formula_len: int = 128,
        output_pooling: str = "attention",
    ) -> None:
        super(FoundationModel, self).__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if output_pooling not in {"attention", "average", "last"}:
            raise ValueError(
                "output_pooling must be one of 'attention', 'average' or 'last', "
                f"got {output_pooling!r}."
            )

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_pooling = output_pooling
        self.formula_positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_formula_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        if output_pooling == "attention":
            self.output_attention_score = nn.Linear(d_model, 1)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        data_embedding: torch.Tensor,
        eq_embedding: torch.Tensor,
        *,
        data_padding_mask: torch.Tensor | None = None,
        eq_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict logits for the next formula symbol.

        Args:
            data_embedding: Embedded data tokens with shape
              ``(batch, data_len, d_model)``.
            eq_embedding: Embedded prefix formula tokens with shape
              ``(batch, formula_len, d_model)``.
            data_padding_mask: Optional bool mask with shape
              ``(batch, data_len)``. ``True`` marks padded data tokens.
            eq_padding_mask: Optional bool mask with shape
              ``(batch, formula_len)``. ``True`` marks padded formula tokens.
        Returns:
            Logits for the next formula symbol with shape ``(batch, vocab_size)``.
        """
        memory = self.encoder(
            data_embedding,
            src_key_padding_mask=data_padding_mask,
        )

        tgt = self.formula_positional_encoding(eq_embedding)
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=eq_padding_mask,
            memory_key_padding_mask=data_padding_mask,
        )

        if self.output_pooling == "last":
            if eq_padding_mask is None:
                pooled = decoded[:, -1, :]
            else:
                lengths = (~eq_padding_mask).sum(dim=1)
                batch_idx = torch.arange(decoded.shape[0], device=decoded.device)
                pooled = decoded[batch_idx, lengths.to(decoded.device) - 1, :]
        elif self.output_pooling == "average":
            if eq_padding_mask is None:
                pooled = decoded.mean(dim=1)
            else:
                valid_mask = ~eq_padding_mask
                lengths = valid_mask.sum(dim=1, keepdim=True)
                pooled = (decoded * valid_mask.unsqueeze(-1)).sum(dim=1) / lengths.to(decoded.dtype)
        elif self.output_pooling == "attention":
            score = self.output_attention_score(decoded).squeeze(-1)
            if eq_padding_mask is not None:
                score = score.masked_fill(eq_padding_mask, float("-inf"))
            weight = torch.softmax(score, dim=1).unsqueeze(-1)
            pooled = (decoded * weight).sum(dim=1)
        else:
            raise ValueError(f"Invalid output_pooling: {self.output_pooling!r}")

        return self.output_projection(pooled)
