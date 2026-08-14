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
    assert result["metrics"]["rmse"] < 1e-9
    assert result["is_candidate"] is True
    assert result["heuristic_exponent_std_across_subset_refits"]["x1"] < 1e-10
    assert result["metrics"]["valid_sample_ratio"] == 1.0


def test_rejects_sign_changing_target():
    x = np.linspace(1, 2, 20)
    y = np.linspace(-1, 1, 20)
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y")()
    assert result.ok is False


def test_reports_domain_exclusions():
    x = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.abs(x) ** 2
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y").execute()
    assert result["metrics"]["valid_sample_ratio"] == 0.6
    assert result["excluded_finite_ranges"]["x"] == [-2.0, -1.0]


def test_rejects_distant_exponent_snap():
    x = np.linspace(0.5, 4, 300)
    y = 3 * x**1.137
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y").execute(
        snap_exponents=True, max_denominator=2, snap_tolerance=0.03
    )
    assert result["simple_exponent_check"]["rounding_applied"] is False
    assert abs(result["exponents"]["x"] - 1.137) < 1e-10


def test_accepts_exact_simple_fraction_snap():
    x = np.linspace(0.5, 4, 300)
    y = 3 * x**1.5
    result = PowerLawFitTool(data={"x": x, "y": y}, target="y").execute(
        snap_exponents=True, max_denominator=4, snap_tolerance=0.03,
    )
    assert result["simple_exponent_check"]["rounding_applied"] is True
    assert result["exponents"]["x"] == 1.5
