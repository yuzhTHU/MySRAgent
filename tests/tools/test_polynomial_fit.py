# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""PolynomialFitTool tests."""

from __future__ import annotations

import numpy as np

from sr_agent.tools.polynomial_fit import PolynomialFitTool


def make_tool(x: dict[str, np.ndarray], y: np.ndarray) -> PolynomialFitTool:
    return PolynomialFitTool(data=x | {"y": y}, target="y")


class TestPolynomialFitTool:
    def setup_method(self):
        x = np.arange(1.0, 11.0)
        y = 2 * x + 1
        self.tool = make_tool({"x": x}, y)

    def test_linear_fit_returns_new_contract(self):
        result = self.tool.execute(max_degree=1)

        assert set(result.keys()) == {"formula", "metrics", "is_candidate", "exceptions"}
        assert isinstance(result["formula"], str)
        assert isinstance(result["is_candidate"], bool)
        assert result["exceptions"] == []
        assert result["metrics"]["r2"] > 0.99
        assert result["metrics"]["rmse"] < 1e-8
        assert "adjusted_r2" in result["metrics"]
        assert "aic" in result["metrics"]
        assert "bic" in result["metrics"]

    def test_quadratic_fit_quality(self):
        x = np.linspace(-5, 5, 50)
        y = x**2 + 2 * x + 1
        result = make_tool({"x": x}, y).execute(max_degree=2)

        assert result["metrics"]["r2"] > 0.99
        assert result["metrics"]["rmse"] < 1e-8
        assert "x" in result["formula"]

    def test_multivariate_interaction_can_fit(self):
        n = 100
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 2 * x1 + 3 * x2 + 1.5 * x1 * x2

        result = make_tool({"x1": x1, "x2": x2}, y).execute(
            max_degree=2,
            include_interactions=True,
        )

        assert result["metrics"]["r2"] > 0.99

    def test_no_interactions_records_worse_fit_for_interaction_target(self):
        n = 100
        rng = np.random.default_rng(1)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = x1 * x2
        tool = make_tool({"x1": x1, "x2": x2}, y)

        with_interactions = tool.execute(max_degree=2, include_interactions=True)
        without_interactions = tool.execute(max_degree=2, include_interactions=False)

        assert with_interactions["metrics"]["r2"] > without_interactions["metrics"]["r2"]

    def test_x_vars_subset(self):
        n = 100
        rng = np.random.default_rng(2)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        y = 2 * x1 + 3 * x2

        result = make_tool({"x1": x1, "x2": x2, "x3": x3}, y).execute(
            x=["x1", "x2"],
            max_degree=1,
        )

        assert result["metrics"]["r2"] > 0.99
        assert "x3" not in result["formula"]

    def test_expression_x_var(self):
        x = np.linspace(1, 5, 20)
        y = 3 * x**2 + 1

        result = make_tool({"x": x}, y).execute(x=["x**2"], max_degree=1)

        assert result["metrics"]["r2"] > 0.99
        assert "x ** 2" in result["formula"] or "x**2" in result["formula"]

    def test_invalid_x_vars_raise_when_no_valid_inputs(self):
        result = self.tool(x=["missing"], max_degree=1)

        assert result.ok is False
        assert "No valid input variables" in result.result_str

    def test_metadata_exists(self):
        assert self.tool.metadata is not None
        assert self.tool.metadata.name == "polynomial_fit"
        assert "polynomial" in self.tool.metadata.description.lower()
