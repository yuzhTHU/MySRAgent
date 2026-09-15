from __future__ import annotations

import numpy as np

from sr_agent.tools.rational_fit import RationalFitTool


def test_recovers_low_degree_rational_function():
    x = np.linspace(-0.5, 2, 300)
    y = (1 + 2 * x) / (1 + 0.4 * x)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        numerator_degree=1, denominator_degree=1
    )
    assert result["data_split_results"]["train"]["metrics"]["rmse"] < 1e-9
    assert not result["denominator_safety_on_observed_samples"]["crosses_zero"]
    assert result["is_candidate"] is True
    assert "Exception:" not in RationalFitTool.format_result_dict(result)


def test_supports_transformed_feature():
    x = np.linspace(0.2, 2, 200)
    y = 1 / (1 + x**2)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        x=["x**2"], numerator_degree=0, denominator_degree=1
    )
    assert result["data_split_results"]["train"]["metrics"]["rmse"] < 1e-9


def test_degree_grid_returns_holdout_ranked_candidates_and_refits():
    x = np.linspace(-0.5, 2, 500)
    y = (1 + 2 * x) / (1 + 0.4 * x)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        numerator_degrees=[0, 1, 2], denominator_degrees=[0, 1], top_k=4
    )
    assert result["selected_polynomial_degrees"]["denominator_degree"] == 1
    assert "near_zero_counts" in result["denominator_safety_on_observed_samples"]
    assert "Held-out RMSE" not in RationalFitTool.format_result_dict(result)
    assert "Selected structure" not in RationalFitTool.format_result_dict(result)


def test_error_table_uses_requested_expressions_and_top_ten_samples():
    a = np.linspace(0.2, 1.2, 40)
    d_a_dt = 0.3 + 0.1 * a
    tool = RationalFitTool(data={"A": a, "dA_dt": d_a_dt}, target="dA_dt")
    result = tool.execute(
        x=["A", "A+dA_dt"], numerator_degree=1, denominator_degree=0,
    )
    samples = result["data_split_results"]["train"]["diagnostics"]["worst_samples"]
    assert len(samples) == 10
    assert result["analyzed_input_expressions"] == ["A", "A+dA_dt"]
    assert samples[0]["row"]["A+dA_dt"] == a[samples[0]["index"]] + d_a_dt[samples[0]["index"]]
    assert "(A | A+dA_dt | dA_dt | residual)" in tool.format_result_dict(result)


def test_denominator_crossing_is_reported_only_when_observed_signs_cross():
    x = np.linspace(-1, 1, 101)
    x = x[~np.isclose(x, 0.5)]
    y = (1 + x) / (1 - 2 * x)
    tool = RationalFitTool(data={"x": x, "y": y}, target="y")
    result = tool.execute(x=["x"], numerator_degree=1, denominator_degree=1)
    text = tool.format_result_dict(result)

    assert result["denominator_safety_on_observed_samples"]["crosses_zero"]
    assert result["linearized_design_matrix"]["rank"] == 3
    denominator_formula = result["denominator_safety_on_observed_samples"]["formula"]
    assert f"Denominator ({denominator_formula}) crosses or reaches zero on training samples" in text
    for threshold in ("1e-8", "1e-6", "1e-4", "1e-2"):
        assert f"fraction(|denominator| < {threshold})=" in text
    assert "Rank[" not in text


def test_denominator_crossing_and_rank_deficiency_are_reported_together():
    x = np.linspace(-1, 1, 101)
    x = x[~np.isclose(x, 0.5)]
    y = (1 + x) / (1 - 2 * x)
    tool = RationalFitTool(data={"x": x, "z": 2 * x, "y": y}, target="y")
    result = tool.execute(x=["x", "z"], numerator_degree=1, denominator_degree=1)
    text = tool.format_result_dict(result)

    assert result["denominator_safety_on_observed_samples"]["crosses_zero"]
    matrix = result["linearized_design_matrix"]
    assert matrix["rank"] < matrix["columns"]
    assert matrix["column_labels"] == ["1", "z", "x", "-y*z", "-y*x"]
    assert "    Linearized design matrix is rank deficient.\n" in text
    assert "        Rank[1, z, x, -y*z, -y*x] = 3 < 5." in text
    denominator_formula = result["denominator_safety_on_observed_samples"]["formula"]
    assert f"    Denominator ({denominator_formula}) crosses or reaches zero" in text
    assert text.index("Rank[1, z, x") < text.index("Denominator (")
