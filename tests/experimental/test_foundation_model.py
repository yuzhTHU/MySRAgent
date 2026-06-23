import argparse

import pytest

torch = pytest.importorskip("torch")

from experimental.nn_tools.models import FoundationModel


@pytest.mark.parametrize("output_pooling", ["attention", "average", "last"])
def test_foundation_model_predicts_next_symbol_logits(output_pooling):
    args = argparse.Namespace(
        d_model=16,
        vocab_size=12,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_formula_len=16,
        output_pooling=output_pooling,
    )
    model = FoundationModel(args)
    data_embedding = torch.randn(2, 5, 16)
    partial_equation_embedding = torch.randn(2, 3, 16)
    partial_equation_padding_mask = torch.tensor([
        [False, False, False],
        [False, False, True],
    ])

    logits = model(
        data_embedding,
        partial_equation_embedding,
        eq_padding_mask=partial_equation_padding_mask,
    )

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
