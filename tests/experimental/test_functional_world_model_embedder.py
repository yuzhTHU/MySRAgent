import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experimental.nn_tools.models.embedder import (
    DataEmbedder,
    EquationEmbedder,
    FloatEmbedder,
)


def test_float_embedder_returns_one_embedding_per_float():
    embedder = FloatEmbedder(d_model=8, mantissa_digits=2, min_exponent=-3, max_exponent=3)

    embedding = embedder(torch.tensor([1.23, -0.01, 0.0, float("inf"), float("-inf"), float("nan")]))

    assert embedding.shape == (6, 8)
    assert torch.isfinite(embedding).all()


def test_data_embedder_preserves_array_length_after_row_pooling():
    shared_float_embedder = FloatEmbedder(d_model=8)
    embedder = DataEmbedder(
        d_model=8,
        float_embedder=shared_float_embedder,
        pooling="attention",
    )
    data_dict = {
        "x": np.linspace(0.0, 1.0, 5),
        "y": np.linspace(1.0, 2.0, 5),
    }

    embedding = embedder(data_dict, ["x", "y"])

    assert embedding.shape == (5, 8)
    assert torch.isfinite(embedding).all()


def test_data_embedder_rejects_mismatched_lengths():
    embedder = DataEmbedder(d_model=8)
    data_dict = {
        "x": np.linspace(0.0, 1.0, 5),
        "y": np.linspace(1.0, 2.0, 4),
    }

    with pytest.raises(ValueError, match="same length"):
        embedder(data_dict, ["x", "y"])


def test_equation_embedder_tokenizes_vectorizes_and_embeds_postorder_nd2py_symbols():
    nd = pytest.importorskip("nd2py")
    embedder = EquationEmbedder(d_model=8)
    eq = nd.parse("sin(x)+3.14*y")

    tokens = embedder.tokenize(eq)
    indices = embedder.vectorize(tokens, variables=["x", "y"])
    embedding = embedder(token=tokens, variables=["x", "y"])

    assert tokens == ["x", "Sin", 3.14, "y", "Mul", "Add"]
    assert indices == [
        embedder.variable_ids[0],
        embedder.token2index["Sin"],
        3.14,
        embedder.variable_ids[1],
        embedder.token2index["Mul"],
        embedder.token2index["Add"],
    ]
    assert embedding.shape == (6, 8)
    assert torch.isfinite(embedding).all()


def test_equation_embedder_batches_variable_length_equations():
    nd = pytest.importorskip("nd2py")
    embedder = EquationEmbedder(d_model=8)
    eq_list = [
        nd.parse("x+y"),
        nd.parse("sin(x)+3.14*y"),
    ]

    embedding = embedder(eq=eq_list, variables=["x", "y"])

    assert embedding.shape == (2, 6, 8)
    assert torch.all(embedding[0, 3:] == 0)
    assert torch.isfinite(embedding).all()


def test_equation_embedder_accepts_custom_operands():
    nd = pytest.importorskip("nd2py")
    embedder = EquationEmbedder(d_model=8, operands=[nd.Add])

    tokens = embedder.tokenize(nd.parse("x+y"))
    indices = embedder.vectorize(tokens, variables=["x", "y"])
    embedding = embedder(index=indices)

    assert len(embedder.operands) == 1
    assert embedder.pad_token_id == 0
    assert embedder.unk_token_id == 1
    assert embedder.symbol_embedding.padding_idx == embedder.pad_token_id
    assert indices[0] == embedder.variable_ids[0]
    assert indices[1] == embedder.variable_ids[1]
    assert embedder.index2token[indices[-1]] == "Add"
    assert embedder.operand_to_id[nd.Add] > embedder.variable_ids[-1]
    assert embedder.symbol_embedding.num_embeddings == (
        2 + embedder.max_variables + len(embedder.operands)
    )
    assert embedding.shape == (3, 8)
