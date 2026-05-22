import pytest

torch = pytest.importorskip("torch")

from experimental.nn_tools.models import PropertyPredictor


def _build_model(shared_decoder_mlp):
    return PropertyPredictor(
        {
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
        },
        properties=[
            {"name": "monotonicity", "scope": "variable", "task": "ternary", "target": "x0"},
            {"name": "interaction", "scope": "variable_pair", "task": "binary", "target": ("x0", "x1")},
            {"name": "noise_level", "scope": "target", "task": "regression"},
        ],
        shared_decoder_mlp=shared_decoder_mlp,
        mlp_hidden_dims=(12,),
    )


@pytest.mark.parametrize("shared_decoder_mlp", [False, True])
def test_property_predictor_outputs_configured_properties(shared_decoder_mlp):
    model = _build_model(shared_decoder_mlp)
    data_embedding = torch.randn(2, 5, 16)
    data_padding_mask = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, True, True],
    ])

    outputs = model(data_embedding, data_padding_mask=data_padding_mask)

    assert set(outputs) == {"monotonicity", "interaction", "noise_level"}
    assert outputs["monotonicity"].shape == (2, 3)
    assert outputs["interaction"].shape == (2, 2)
    assert outputs["noise_level"].shape == (2, 1)
    assert all(torch.isfinite(value).all() for value in outputs.values())
