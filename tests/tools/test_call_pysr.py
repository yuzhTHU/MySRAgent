# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""PySRTool tests."""

from __future__ import annotations

import numpy as np

from sr_agent.tools.call_pysr import PySRTool


def make_tool(x: dict[str, np.ndarray], y: np.ndarray) -> PySRTool:
    return PySRTool(data=x | {"y": y}, target="y")


class TestPySRTool:
    def test_execute_restores_feature_names_before_evaluation(self, monkeypatch):
        x = np.linspace(-2.0, 2.0, 20)
        y = 2 * x + 1
        tool = make_tool({"x": x}, y)

        def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
            assert x_names == ["x1"]
            return "2*x1 + 1", [{"formula": "2*x1 + 1", "loss": 0.0, "complexity": 3}], 3

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)

        result = tool.execute(binary_operators=["+", "*"], unary_operators=[])

        assert result["formula"] == "2 * x + 1"
        assert result["pareto_front"][0]["formula"] == "2 * x + 1"
        assert result["metrics"]["mse"] < 1e-12
        assert result["metrics"]["r2"] == 1.0
        assert result["is_candidate"] is True
        assert result["exceptions"] == []

    def test_execute_supports_expression_features(self, monkeypatch):
        x = np.linspace(-3.0, 3.0, 30)
        y = 4 * x**2 + 0.5
        tool = make_tool({"x": x}, y)

        def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
            np.testing.assert_allclose(X[:, 0], x**2)
            return "4*x1 + 0.5", [], 2

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)

        result = tool.execute(
            binary_operators=["+", "*"],
            unary_operators=[],
            x=["x**2"],
        )

        assert result["formula"] == "4 * x ** 2 + 0.5"
        assert result["metrics"]["mse"] < 1e-12

    def test_restore_feature_names_does_not_rewrite_inserted_expressions(self, monkeypatch):
        x2 = np.linspace(0.0, 3.0, 12)
        z = np.linspace(1.0, 4.0, 12)
        y = x2 + z
        tool = make_tool({"x2": x2, "z": z}, y)

        def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
            return "x1 + x2", [], 2

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)

        result = tool.execute(binary_operators=["+"], unary_operators=[], x=["x2", "z"])

        assert result["formula"] == "x2 + z"
        assert result["metrics"]["mse"] < 1e-12

    def test_execute_clamps_timeout_and_subsamples(self, monkeypatch):
        x = np.arange(20.0)
        y = x + 1
        tool = make_tool({"x": x}, y)
        seen = {}

        def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
            seen["shape"] = X.shape
            seen["timeout"] = timeout
            seen["maxsize"] = maxsize
            return "x1 + 1", [], 2

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)

        result = tool.execute(
            binary_operators=["+"],
            unary_operators=[],
            timeout=999,
            maxsize=7,
            max_samples=5,
        )

        assert seen == {"shape": (5, 1), "timeout": 120, "maxsize": 7}
        assert result["config"]["timeout"] == 120
        assert result["metrics"]["mse"] < 1e-12

    def test_execute_uses_gplearn_fallback_when_pysr_fails(self, monkeypatch):
        x = np.linspace(0.0, 5.0, 10)
        y = x + 2
        tool = make_tool({"x": x}, y)

        def fake_run_pysr(self, *args, **kwargs):
            raise RuntimeError("julia unavailable")

        def fake_fallback(self, X, y_fit, x_names, binary_ops, unary_ops):
            return "x1 + 2"

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)
        monkeypatch.setattr(PySRTool, "_run_gplearn_fallback", fake_fallback)

        result = tool.execute(binary_operators=["+"], unary_operators=[])

        assert result["method"] == "gplearn"
        assert result["formula"] == "x + 2"
        assert result["metrics"]["mse"] < 1e-12
        assert any("PySR failed" in item for item in result["exceptions"])

    def test_invalid_x_vars_raise_when_no_valid_inputs(self):
        x = np.arange(5.0)
        y = x + 1
        result = make_tool({"x": x}, y)(x=["missing"], binary_operators=["+"], unary_operators=[])

        assert result.ok is False
        assert "No valid input variables" in result.result_str

    def test_quoted_y_parameter_is_stripped(self, monkeypatch):
        """LLM sometimes passes y with extra quotes like '"omega"'."""
        x = np.linspace(-2.0, 2.0, 20)
        omega = 2 * x + 1
        tool = PySRTool(data={"x": x, "omega": omega}, target="omega")

        def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
            return "2*x1 + 1", [{"formula": "2*x1 + 1", "loss": 0.0, "complexity": 3}], 3

        monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)

        result = tool.execute(binary_operators=["+", "*"], unary_operators=[], y='"omega"')
        assert result["metrics"]["mse"] < 1e-12
        assert result["is_candidate"] is True

    def test_metadata_exists(self):
        x = np.array([1.0])
        tool = make_tool({"x": x}, x)

        assert tool.metadata is not None
        assert tool.metadata.name == "call_pysr"
        assert "pysr" in tool.metadata.description.lower()
