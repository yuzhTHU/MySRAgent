from __future__ import annotations

import numpy as np

from sr_agent.tools.power_law_fit import PowerLawFitTool


def test_recovers_multivariate_power_law():
    rng = np.random.default_rng(2)
    x1 = rng.uniform(0.5, 3, 300)
    x2 = rng.uniform(0.5, 3, 300)
    y = 2.5 * x1**2 * x2**-0.5
    result = PowerLawFitTool(data={"x1": x1, "x2": x2, "y": y}, target="y").execute()
    assert abs(result["exponents"]["x1"] - 2) < 1e-10
    assert abs(result["exponents"]["x2"] + 0.5) < 1e-10
    assert result["data_split_results"]["train"]["metrics"]["rmse"] < 1e-9
    assert result["is_candidate"] is True
    lower, upper = result["exponent_confidence_intervals"]["x1"]
    assert abs(lower - 2) < 1e-10
    assert abs(upper - 2) < 1e-10
    assert result["data_split_results"]["train"]["metrics"]["valid_sample_ratio"] == 1.0
    text = PowerLawFitTool.format_result_dict(result)
    assert "Training-set log-space RMSE" not in text
    assert "Samples satisfying the nonzero-target" not in text
    assert "Exponent stability (Divide the 300 training samples into 5 folds, use 4 of the folds to fit the exponents, and repeat the process 5 times):" in text
    assert "x1: 95% CI=[" in text


def test_ci_endpoints_add_digits_only_when_needed():
    format_ci = PowerLawFitTool._format_ci_endpoints
    assert format_ci(1.4, 1.5) == ("1.40", "1.50")
    assert format_ci(1.0001, 1.0002) == ("1.0001", "1.0002")
    assert format_ci(1.0000001, 1.0000002) == ("1.00000", "1.00000")


def test_rejects_sign_changing_target():
    x = np.linspace(1, 2, 20)
    y = np.linspace(-1, 1, 20)
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y")()
    assert result.ok is True
    assert result.result == {
        "exceptions": ["y has 10/20 (50.0%) non-positive finite values"],
        "domain_inapplicable": True,
    }
    assert result.result_str == (
        "Power-law fitting not applicable:\n"
        "    y has 10/20 (50.0%) non-positive finite values\n"
        "power_law_fit requires every x and y value to be finite and strictly positive."
    )


def test_reports_nonpositive_fraction_for_every_affected_variable():
    x1 = np.array([-2.0, 0.0, 1.0, 2.0, 3.0])
    x2 = np.array([1.0, 2.0, 0.0, 2.0, 3.0])
    y = np.array([1.0, -1.0, 2.0, 2.0, 3.0])
    result = PowerLawFitTool(data={"x1": x1, "x2": x2, "y": y}, target="y")()
    assert result.ok is True
    assert len(result.result["exceptions"]) == 3
    assert "x1 has 2/5 (40.0%)" in result.result["exceptions"][0]
    assert "x2 has 1/5 (20.0%)" in result.result["exceptions"][1]
    assert "y has 1/5 (20.0%)" in result.result["exceptions"][2]
    assert "Best fitted power-law formula" not in result.result_str


def test_reports_nonfinite_values_and_invalid_y_type():
    x = np.array([1.0, np.nan, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    tool = PowerLawFitTool(data={"x": x, "y": y}, target="y")
    result = tool.execute()
    assert "1/3 (33.3%) non-finite values" in result["exceptions"][0]
    assert "y must be a string" in tool.execute(y=["y"])["exceptions"][0]


def test_domain_error_lists_only_present_issues_and_nonzero_small_percentages():
    t = np.ones(4000)
    t[0] = 0
    a = np.ones(4000)
    a[:12] = np.nan
    y = np.ones(4000)
    y[:123] = -1
    call = PowerLawFitTool(data={"t": t, "A": a, "dA_dt": y}, target="dA_dt")()
    assert call.result_str == (
        "Power-law fitting not applicable:\n"
        "    t has 1/4000 (0.025%) non-positive finite values\n"
        "    A has 12/4000 (0.3%) non-finite values\n"
        "    dA_dt has 123/4000 (3.08%) non-positive finite values\n"
        "power_law_fit requires every x and y value to be finite and strictly positive."
    )


def test_negative_infinity_is_counted_only_as_nonfinite():
    call = PowerLawFitTool(
        data={"x": np.array([-np.inf, -1.0, 1.0]), "y": np.ones(3)}, target="y"
    )()
    assert call.result["exceptions"] == [
        "x has 1/3 (33.3%) non-positive finite values",
        "x has 1/3 (33.3%) non-finite values",
    ]


def test_rejects_distant_exponent_snap():
    x = np.linspace(0.5, 4, 300)
    y = 3 * x**1.137
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y").execute(
        snap_exponents=True, max_denominator=2, snap_tolerance=0.03
    )
    assert result["simple_exponent_check"]["rounding_applied"] is False
    assert abs(result["exponents"]["x"] - 1.137) < 1e-10


def test_simplified_exponent_check_reports_validation_r2_without_using_it_for_selection():
    x = np.exp(np.linspace(-1, 1, 300))
    validation_x = np.exp(np.linspace(-0.9, 0.9, 100))
    tool = PowerLawFitTool(
        data={"x": x, "y": 3 * x**1.137},
        evaluation_data={"x": validation_x, "y": 3 * validation_x},
        target="y",
    )
    result = tool.execute(
        snap_exponents=True, max_denominator=1, snap_tolerance=0.03
    )
    check = result["simple_exponent_check"]
    assert check["rounding_applied"] is False
    assert check["snapped_validation_r2"] > check["raw_validation_r2"]
    text = tool.format_result_dict(result)
    assert "Simplified exponent check:\n    Direct fitting on 300 samples yields exponents:" in text
    assert "would increase validation-set R2 by" in text.split("Simplified exponent check:", 1)[1]
    assert "the simplified exponents were thus used" not in text


def test_accepts_exact_simple_fraction_snap():
    x = np.linspace(0.5, 4, 300)
    y = 3 * x**1.5
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y").execute(
        snap_exponents=True, max_denominator=4, snap_tolerance=0.03,
    )
    assert result["simple_exponent_check"]["rounding_applied"] is True
    assert result["exponents"]["x"] == 1.5
    assert "the simplified exponents were thus used as the best fitted power-law formula." in PowerLawFitTool.format_result_dict(result)


def test_fractional_exponent_is_visible_in_formula_and_check():
    x = np.linspace(0.5, 3, 300)
    validation_x = np.linspace(0.6, 3.2, 60)
    tool = PowerLawFitTool(
        data={"x": x, "y": x ** (4 / 3)},
        evaluation_data={"x": validation_x, "y": validation_x ** (4 / 3)},
        target="y",
    )
    result = tool.execute(snap_exponents=True)
    assert result["simple_exponent_check"]["rounding_applied"] is True
    assert result["simple_exponent_check"]["snapped_exponent_expressions"] == {"x": "4/3"}
    assert "x ** (4 / 3)" in result["formula"]
    assert "Number(" not in result["formula"]
    assert "replacing them with {'x': '4/3'}" in tool.format_result_dict(result)
    assert result["data_split_results"]["validation"]["metrics"]["r2"] > 0.999999
