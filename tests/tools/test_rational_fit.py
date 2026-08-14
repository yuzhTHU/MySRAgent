from __future__ import annotations

import numpy as np

from sr_agent.tools.rational_fit import RationalFitTool


def test_recovers_low_degree_rational_function():
    x = np.linspace(-0.5, 2, 300)
    y = (1 + 2 * x) / (1 + 0.4 * x)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        numerator_degree=1, denominator_degree=1
    )
    assert result["metrics"]["rmse"] < 1e-9
    assert result["denominator_safety_on_observed_samples"]["minimum_absolute_denominator"] > 0
    assert result["is_candidate"] is True


def test_supports_transformed_feature():
    x = np.linspace(0.2, 2, 200)
    y = 1 / (1 + x**2)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        x=["x**2"], numerator_degree=0, denominator_degree=1
    )
    assert result["metrics"]["rmse"] < 1e-9


def test_degree_grid_returns_holdout_ranked_candidates_and_refits():
    x = np.linspace(-0.5, 2, 500)
    y = (1 + 2 * x) / (1 + 0.4 * x)
    result = RationalFitTool(data={"x": x, "y": y}, target="y").execute(
        numerator_degrees=[0, 1, 2], denominator_degrees=[0, 1], top_k=4
    )
    assert result["selected_polynomial_degrees"]["denominator_degree"] == 1
    assert "first_percentile_absolute_denominator" in result["denominator_safety_on_observed_samples"]
    assert result["heldout_rmse_across_subsamples"]["number_of_subsamples"] == 5
    assert result["heldout_rmse_across_subsamples"]["mean_rmse"] < 1e-8
