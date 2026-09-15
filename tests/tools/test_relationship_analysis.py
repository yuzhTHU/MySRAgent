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
        n_bins=80, n_folds=5
    )
    rel = result["relationships"]["x"]
    assert rel["one_variable_test_r2_mean"] < 0.3
    assert np.isfinite(rel["one_variable_test_r2_std"])


def test_spline_collapse_generalizes_for_smooth_curve():
    x = np.linspace(-2, 2, 500)
    y = x**3 - x
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y").execute(
        collapse_model="spline", n_folds=5
    )
    assert result["relationships"]["x"]["one_variable_test_r2_mean"] > 0.95


def test_formatted_output_reports_methods_and_values_without_shape_advice():
    x = np.linspace(0, 4, 40)
    y = x**2
    tool = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y")
    call = tool(n_bins=4, pairwise=True)
    assert call.ok
    assert call.result["analysis_settings"]["binning"] == "quantile"
    assert call.result["analysis_settings"]["n_folds"] == 5
    assert call.result["relationships"]["x"]["one_variable_test_success_folds"] == 5
    assert call.result["relationships"]["x"]["jointly_finite_sample_count"] == 40
    assert "binned_shape" not in call.result["relationships"]["x"]
    assert "method=quantile; duplicate quantile cut points removed" in call.result_str
    assert "Pearson linear correlation=" in call.result_str
    assert "Spearman rank correlation=" in call.result_str
    assert "One-variable held-out experiment" in call.result_str
    assert "Success folds=5/5" in call.result_str
    assert "(Range of x | samples | y mean | y std | y range)" in call.result_str
    assert " | " in call.result_str
    assert "Pairwise correlations" in call.result_str
    for phrase in ("Try ", "hypothesis", "may indicate", "Heuristic shape clue", "causal effect"):
        assert phrase not in call.result_str


def test_five_folds_each_hold_out_distinct_samples(monkeypatch):
    x = np.arange(50, dtype=float)
    y = x**2 + 1
    observed = []
    original = RelationshipAnalysisTool._fit_predict_1d

    def capture(train_x, train_y, test_x, n_bins, binning, model):
        predicted = original(train_x, train_y, test_x, n_bins, binning, model)
        observed.append((train_x.copy(), test_x.copy(), float(np.mean(train_y)), predicted))
        return predicted

    monkeypatch.setattr(RelationshipAnalysisTool, "_fit_predict_1d", staticmethod(capture))
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y").execute(n_folds=5)
    assert result["relationships"]["x"]["one_variable_test_success_folds"] == 5
    assert len(observed) == 5
    held_out = [set(test) for _, test, _, _ in observed]
    assert all(not (set(train) & set(test)) for train, test, _, _ in observed)
    assert set.union(*held_out) == set(x)
    assert sum(len(test) for test in held_out) == len(x)
    scores = []
    for _, test, train_mean, predicted in observed:
        test_y = test**2 + 1
        scores.append(1 - np.sum((test_y - predicted) ** 2) / np.sum((test_y - train_mean) ** 2))
    rel = result["relationships"]["x"]
    assert rel["one_variable_test_r2_mean"] == np.mean(scores)
    assert rel["one_variable_test_r2_std"] == np.std(scores)


def test_float_display_has_three_significant_digits():
    x = np.linspace(0, 54, 100)
    y = 1.23e-8 * x
    output = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y")().result_str
    assert "x range=[0, 54.0]" in output
    assert "y range=[0, 6.64e-07]" in output


def test_default_five_quantile_bins_merge_duplicate_cut_points():
    x = np.concatenate([np.zeros(600), np.linspace(0, 1, 401)[1:]])
    y = 2 * x + 1
    result = RelationshipAnalysisTool(data={"x": x, "y": y}, target="y")()
    assert result.ok
    assert result.result["analysis_settings"]["n_bins"] == 5
    bins = result.result["relationships"]["x"]["conditional_bins"]
    assert len(bins) == 3
    assert [item["sample_count"] for item in bins] == [600, 200, 200]
    assert "Binning analysis (n_bins=5, method=quantile; duplicate quantile cut points removed)" in result.result_str
    assert "0 <= x < 0.00100 | 600" in result.result_str
