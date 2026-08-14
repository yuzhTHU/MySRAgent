from __future__ import annotations

import numpy as np

from sr_agent.tools.constant_fit import ConstantFitTool


def test_fits_scale_and_recognizes_pi():
    x = np.linspace(1, 4, 100)
    y = np.pi * x
    result = ConstantFitTool(data={"x": x, "y": y}, target="y").execute(eq="x")
    assert abs(result["constant"] - np.pi) < 1e-12
    assert result["metrics"]["rmse"] < 1e-10
    assert result["plausible_simple_constant"]["expression"] == "1*pi"
    assert result["subsample_constant_relative_std"] < 1e-12
    assert result["is_candidate"] is True


def test_eq_as_y_scores_an_invariant():
    x = np.linspace(1, 4, 100)
    y = 7 / x
    result = ConstantFitTool(data={"x": x, "y": y}, target="y").execute(
        eq="x*y", use_eq_as_y=True
    )
    assert abs(result["constant"] - 7) < 1e-12
    assert result["metrics"]["rmse"] < 1e-10
    assert result["mode"] == "eq_as_constant"
    assert result["is_candidate"] is False
    assert result["usable_sample_fraction"] == 1.0


def test_named_constant_is_not_accepted_when_approximation_is_too_rough():
    x = np.linspace(1, 4, 100)
    y = 1.23456789 * x
    result = ConstantFitTool(data={"x": x, "y": y}, target="y").execute(
        eq="x", max_denominator=4, recognition_tolerance=1e-10
    )
    assert result["plausible_simple_constant"] is None
