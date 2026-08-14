from __future__ import annotations

import numpy as np

from sr_agent.tools.relationship_analysis import RelationshipAnalysisTool


def test_relationships_and_conditional_collapse():
    x = np.linspace(0, 10, 200)
    y = 3 * x + 1
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y").execute(n_bins=10)
    rel = result["relationships"]["x"]
    assert rel["pearson"] > 0.999
    assert rel["spearman"] > 0.999
    assert rel["one_variable_test_r2_mean"] > 0.98
    assert len(rel["conditional_bins"]) == 10
    assert "pairwise_correlations" not in result


def test_expressions_and_optional_pairwise_matrix():
    x = np.linspace(1, 5, 50)
    result = RelationshipAnalysisTool(data={"x": x, "y": x**2}, target="y").execute(
        variables=["x", "x**2"], pairwise=True, n_bins=5
    )
    matrix = result["pairwise_correlations"]
    assert matrix["variables"] == ["x", "x**2", "y"]
    assert np.asarray(matrix["pearson"]).shape == (3, 3)
    assert result["relationships"]["x"]["strongest_residual_association_after_one_variable_fit"]["variable"] == "x**2"


def test_invalid_binning_is_wrapped_by_base_tool():
    tool = RelationshipAnalysisTool(data={"x": np.arange(4), "y": np.arange(4)}, target="y")
    result = tool(binning="bad")
    assert result.ok is False
    assert "binning" in result.result_str


def test_validated_collapse_rejects_many_bin_noise_overfit():
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y").execute(
        n_bins=80, n_repeats=5, validation_fraction=0.25
    )
    rel = result["relationships"]["x"]
    assert rel["one_variable_test_r2_mean"] < 0.3
    assert np.isfinite(rel["one_variable_test_r2_std"])


def test_spline_collapse_generalizes_for_smooth_curve():
    x = np.linspace(-2, 2, 500)
    y = x**3 - x
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y").execute(
        collapse_model="spline", n_repeats=3
    )
    assert result["relationships"]["x"]["one_variable_test_r2_mean"] > 0.95
