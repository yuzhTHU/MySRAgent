# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""PolynomialFitTool tests."""

from __future__ import annotations

import numpy as np

from sr_agent.tools.polynomial_fit import PolynomialFitTool


def make_tool(x: dict[str, np.ndarray], y: np.ndarray) -> PolynomialFitTool:
    return PolynomialFitTool(data=x | {"y": y}, target="y")


def train_metrics(result):
    return result["data_split_results"]["train"]["metrics"]


class TestPolynomialFitTool:
    def setup_method(self):
        x = np.arange(1.0, 11.0)
        y = 2 * x + 1
        self.tool = make_tool({"x": x}, y)

    def test_linear_fit_returns_new_contract(self):
        result = self.tool.execute(max_degree=1)
        assert set(result) == {
            "formula", "target_expression", "is_candidate",
            "candidate_ineligibility_reasons", "data_split_results",
            "fit_configuration", "simplification", "exceptions",
        }
        assert result["is_candidate"] is True
        assert result["exceptions"] == []
        assert result["fit_configuration"]["input_features"] == ["x"]
        assert train_metrics(result)["r2"] > 0.99
        assert train_metrics(result)["rmse"] < 1e-8

    def test_constant_term_is_not_rendered_as_coefficient_times_one(self):
        result = self.tool.execute(max_degree=1)
        assert "* 1" not in result["formula"]

    def test_default_simplification_removes_negligible_monomials_and_refits(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 200)
        z = rng.uniform(-2, 2, 200)
        y = 1 + 2 * x + x**2
        result = make_tool({"x": x, "z": z}, y).execute(max_degree=2)

        assert result["simplification"]["enabled"] is True
        assert result["simplification"]["removed_terms"]
        assert "z" not in result["formula"]
        assert train_metrics(result)["rmse"] < 1e-10

    def test_simplification_can_be_disabled(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, 200)
        y = 1 + 2 * x + 1e-7 * x**2
        tool = make_tool({"x": x}, y)

        simplified = tool.execute(max_degree=2)
        unsimplified = tool.execute(max_degree=2, simplify=False)

        assert "x ** 2" not in simplified["formula"]
        assert "x ** 2" in unsimplified["formula"]
        assert unsimplified["simplification"]["enabled"] is False

    def test_quadratic_fit_quality(self):
        x = np.linspace(-5, 5, 50)
        y = x**2 + 2 * x + 1
        result = make_tool({"x": x}, y).execute(max_degree=2)
        assert train_metrics(result)["r2"] > 0.99
        assert train_metrics(result)["rmse"] < 1e-8

    def test_multivariate_interaction_can_fit(self):
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=100)
        x2 = rng.normal(size=100)
        y = 2 * x1 + 3 * x2 + 1.5 * x1 * x2
        result = make_tool({"x1": x1, "x2": x2}, y).execute(
            max_degree=2, include_interactions=True,
        )
        assert train_metrics(result)["r2"] > 0.99

    def test_no_interactions_records_worse_fit_for_interaction_target(self):
        rng = np.random.default_rng(1)
        x1 = rng.normal(size=100)
        x2 = rng.normal(size=100)
        tool = make_tool({"x1": x1, "x2": x2}, x1 * x2)
        with_interactions = tool.execute(max_degree=2, include_interactions=True)
        without_interactions = tool.execute(max_degree=2, include_interactions=False)
        assert train_metrics(with_interactions)["r2"] > train_metrics(without_interactions)["r2"]

    def test_x_vars_subset(self):
        rng = np.random.default_rng(2)
        x1 = rng.normal(size=100)
        x2 = rng.normal(size=100)
        x3 = rng.normal(size=100)
        result = make_tool({"x1": x1, "x2": x2, "x3": x3}, 2 * x1 + 3 * x2).execute(
            x=["x1", "x2"], max_degree=1,
        )
        assert train_metrics(result)["r2"] > 0.99
        assert "x3" not in result["formula"]

    def test_expression_x_var(self):
        x = np.linspace(1, 5, 20)
        result = make_tool({"x": x}, 3 * x**2 + 1).execute(x=["x**2"], max_degree=1)
        assert train_metrics(result)["r2"] > 0.99
        assert "x ** 2" in result["formula"] or "x**2" in result["formula"]

    def test_invalid_x_vars_raise_when_no_valid_inputs(self):
        result = self.tool(x=["missing"], max_degree=1)
        assert result.ok is False
        assert "No valid input variables" in result.result_str

    def test_quoted_y_parameter_is_stripped(self):
        x = np.arange(1.0, 6.0)
        omega = 3 * x + 2
        tool = PolynomialFitTool(data={"x": x, "omega": omega}, target="omega")
        result = tool.execute(y='"omega"', max_degree=1)
        assert train_metrics(result)["r2"] > 0.99
        assert result["is_candidate"] is True

    def test_agent_output_omits_fit_configuration(self):
        result = self.tool(max_degree=1)
        assert result.ok is True
        assert "Fit configuration:" not in result.result_str

    def test_metadata_exists(self):
        assert self.tool.metadata is not None
        assert self.tool.metadata.name == "polynomial_fit"
        assert "polynomial" in self.tool.metadata.description.lower()
