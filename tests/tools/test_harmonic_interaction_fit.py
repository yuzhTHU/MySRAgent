from __future__ import annotations

import numpy as np
import pytest

from sr_agent.tools import BaseTool, HarmonicInteractionFitTool


def test_recovers_polynomial_plus_sinusoidal_interaction():
    rng = np.random.default_rng(43)
    t = rng.uniform(1, 54, 2000)
    p = rng.uniform(1, 60, 2000)
    y = 0.7 * p - 0.01 * p**2 + 0.88 * p * np.sin(0.567 * t)
    tool = HarmonicInteractionFitTool(
        data={"t": t[:1600], "p": p[:1600], "y": y[:1600]},
        evaluation_data={"t": t[1600:], "p": p[1600:], "y": y[1600:]},
        target="y",
    )

    result = tool.execute(carrier="p", oscillator="t", show_diagnostics=False)

    assert result["is_candidate"] is True
    assert result["data_split_results"]["validation"]["metrics"]["mse"] < 1e-9
    assert abs(result["fit_configuration"]["frequency"] - 0.567) < 1e-5
    assert "sin" in result["formula"]
    assert BaseTool.create("harmonic_interaction_fit", create_instance=False) is HarmonicInteractionFitTool


def test_rejects_constant_oscillator():
    x = np.linspace(1, 3, 100)
    tool = HarmonicInteractionFitTool(
        data={"t": np.ones_like(x), "x": x, "y": x}, target="y"
    )
    with pytest.raises(ValueError, match="Oscillator must vary"):
        tool.execute(carrier="x", oscillator="t")
