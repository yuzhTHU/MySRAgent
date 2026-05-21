# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""SINDyTool tests."""

from __future__ import annotations

import numpy as np

from sr_agent.tools.call_sindy import SINDyTool


def make_tool(x: dict[str, np.ndarray], y: np.ndarray) -> SINDyTool:
    return SINDyTool(data=x | {"y": y}, target="y")


class TestSINDyTool:
    def test_execute_restores_feature_names_before_evaluation(self, monkeypatch):
        x = np.linspace(-2.0, 2.0, 20)
        y = 2 * x + 1
        tool = make_tool({"x": x}, y)

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            assert x_names == ["x1"]
            return "2*x1 + 1"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(poly_degree=1)

        assert result["formula"] == "2 * x + 1"
        assert result["metrics"]["mse"] < 1e-12
        assert result["metrics"]["r2"] == 1.0
        assert result["is_candidate"] is True
        assert result["exceptions"] == []

    def test_execute_supports_expression_features(self, monkeypatch):
        x = np.linspace(-3.0, 3.0, 30)
        y = 4 * x**2 + 0.5
        tool = make_tool({"x": x}, y)

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            np.testing.assert_allclose(X[:, 0], x**2)
            return "4*x1 + 0.5"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(x=["x**2"], poly_degree=1)

        assert result["formula"] == "4 * x ** 2 + 0.5"
        assert result["metrics"]["mse"] < 1e-12

    def test_restore_feature_names_does_not_rewrite_inserted_expressions(self, monkeypatch):
        x2 = np.linspace(0.0, 3.0, 12)
        z = np.linspace(1.0, 4.0, 12)
        y = x2 + z
        tool = make_tool({"x2": x2, "z": z}, y)

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            return "x1 + x2"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(x=["x2", "z"])

        assert result["formula"] == "x2 + z"
        assert result["metrics"]["mse"] < 1e-12

    def test_execute_normalizes_square_in_final_formula(self, monkeypatch):
        x = np.linspace(-2.0, 2.0, 20)
        z = np.linspace(-1.0, 1.0, 20)
        y = (x - z) ** 2
        tool = make_tool({"x": x, "z": z}, y)

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            return "square(x1 - x2)"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(x=["x", "z"])

        assert result["formula"] == "(x - z) ** 2"
        assert result["metrics"]["mse"] < 1e-12

    def test_execute_normalizes_nested_square_in_final_formula(self, monkeypatch):
        x = np.linspace(-2.0, 2.0, 20)
        y = np.sin(x) ** 2
        tool = make_tool({"x": x}, y)

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            return "square(sin(x1))"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(x=["x"])

        assert result["formula"] == "sin(x) ** 2"
        assert result["metrics"]["mse"] < 1e-12

    def test_execute_clamps_config_and_subsamples(self, monkeypatch):
        x = np.arange(20.0)
        y = x + 1
        tool = make_tool({"x": x}, y)
        seen = {}

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            seen["shape"] = X.shape
            seen["poly_degree"] = poly_degree
            seen["include_trig"] = include_trig
            seen["threshold"] = threshold
            return "x1 + 1"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(
            poly_degree=99,
            include_trig=True,
            threshold=0.0,
            max_samples=5,
        )

        assert seen == {
            "shape": (5, 1),
            "poly_degree": 5,
            "include_trig": True,
            "threshold": 0.01,
        }
        assert result["config"]["poly_degree"] == 5
        assert result["config"]["threshold"] == 0.01
        assert result["config"]["max_samples"] == 5
        assert result["metrics"]["mse"] < 1e-12

    def test_sindy_failure_returns_inf_metrics(self, monkeypatch):
        x = np.linspace(0.0, 5.0, 10)
        y = x + 2
        tool = make_tool({"x": x}, y)

        def fake_run_sindy(self, *args, **kwargs):
            raise RuntimeError("pysindy unavailable")

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute()

        assert result["formula"] == "0"
        assert result["metrics"]["mse"] == float("inf")
        assert result["is_candidate"] is False
        assert any("SINDy fitting failed" in item for item in result["exceptions"])

    def test_invalid_x_vars_raise_when_no_valid_inputs(self):
        x = np.arange(5.0)
        y = x + 1
        result = make_tool({"x": x}, y)(x=["missing"])

        assert result.ok is False
        assert "No valid input variables" in result.result_str

    def test_clean_formula_parses_sindy_terms(self):
        tool = make_tool({"x": np.array([1.0])}, np.array([1.0]))

        formula = tool._clean_formula("1.000 x1 + -2.500 x1^2 + 0.000 x1", ["x1"])

        assert formula == "x1 - 2.5*x1**2"

    def test_quoted_y_parameter_is_stripped(self, monkeypatch):
        """LLM sometimes passes y with extra quotes like '"omega"'."""
        x = np.linspace(-2.0, 2.0, 20)
        omega = 2 * x + 1
        tool = SINDyTool(data={"x": x, "omega": omega}, target="omega")

        def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
            return "2*x1 + 1"

        monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)

        result = tool.execute(y='"omega"')
        assert result["metrics"]["mse"] < 1e-12
        assert result["is_candidate"] is True

    def test_metadata_exists(self):
        x = np.array([1.0])
        tool = make_tool({"x": x}, x)

        assert tool.metadata is not None
        assert tool.metadata.name == "call_sindy"
        assert "sindy" in tool.metadata.description.lower()
