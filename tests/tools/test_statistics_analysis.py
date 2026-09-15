"""StatisticsTool 的单元测试。"""

from __future__ import annotations

import numpy as np

from sr_agent.tools.statistics_analysis import StatisticsTool


class TestStatisticsToolMetadata:
    def test_metadata_is_inferred_from_execute_docstring(self):
        assert StatisticsTool.metadata.name == "statistics_analysis"
        assert StatisticsTool.metadata.description == "Execute statistical analysis."

    def test_parameters_schema_is_inferred_from_execute_signature_and_docstring(self):
        assert StatisticsTool.metadata.parameters == {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'List of variable names to analyze, e.g., ["x1", "x2", "y"].\n'
                        "Use all variables (including the target variable) by default.\n"
                        'Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2", "sin(y+x1)"].'
                    ),
                    "default": None,
                },
                "n_bins": {
                    "type": "integer",
                    "description": "Number of equal-width histogram bins used to summarize each distribution (1-100).",
                    "default": 10,
                },
                "near_zero_threshold": {
                    "type": "number",
                    "description": (
                        "First absolute-value threshold used to count near-zero samples.\n"
                        "The output also reports thresholds 1e-6 and 1e-4."
                    ),
                    "default": 1e-8,
                },
            },
            "required": [],
        }

    def test_to_dict_exports_openrouter_tool_schema(self):
        assert StatisticsTool.to_dict() == {
            "type": "function",
            "function": {
                "name": "statistics_analysis",
                "description": "Execute statistical analysis.",
                "parameters": StatisticsTool.metadata.parameters,
            },
        }


class TestStatisticsToolExecution:
    def setup_method(self):
        self.data = {
            "x1": np.array([1.0, 2.0, 3.0, 4.0]),
            "x2": np.array([10.0, 20.0, 30.0, 40.0]),
            "y": np.array([2.0, 4.0, 6.0, 8.0]),
        }
        self.tool = StatisticsTool(data=self.data)

    def test_execute_analyzes_all_variables_by_default(self):
        result = self.tool.execute()

        assert set(result.keys()) == {"statistics", "config", "exceptions"}
        assert set(result["statistics"].keys()) == {"x1", "x2", "y"}
        assert result["exceptions"] == []

    def test_execute_analyzes_selected_variables(self):
        result = self.tool.execute(variables=["x2", "y"])

        assert set(result["statistics"].keys()) == {"x2", "y"}
        assert result["statistics"]["x2"]["mean"] == 25.0
        assert result["statistics"]["y"]["max"] == 8.0

    def test_execute_preserves_requested_variable_order(self):
        result = self.tool.execute(variables=["y", "x1"])

        assert list(result["statistics"].keys()) == ["y", "x1"]

    def test_execute_records_unknown_variable_exception(self):
        result = self.tool(variables=["missing"])

        assert result.ok is True
        assert result.result["statistics"] == {}
        assert result.result["exceptions"]
        assert "missing" in result.result_str

    def test_call_wraps_successful_result(self):
        result = self.tool(variables=["x1"])

        assert result.ok is True
        assert set(result.result["statistics"].keys()) == {"x1"}
        assert "x1 (finite samples=4/4; finite ratio=100%):" in result.result_str
        assert result.meta_data["tool"] == "statistics_analysis"

    def test_multiple_invariant_expressions_include_nearly_constant_one(self):
        x = np.linspace(1, 2, 10)
        call = StatisticsTool(
            data={"x1": x, "x2": 2 * x, "y": 3 / x}, target="y"
        )(variables=["y*x1", "y*x1/x2"])
        assert call.ok is True
        assert call.result["exceptions"] == []
        assert list(call.result["statistics"]) == ["y*x1", "y*x1/x2"]
        assert call.result["statistics"]["y*x1"]["distribution"]["n_bins"] == 1
        assert "near-constant values combined" in call.result_str


class TestStatisticsToolStats:
    def test_get_stats_computes_all_metrics(self):
        tool = StatisticsTool(data={})
        stats = tool.get_stats(np.array([1.0, 2.0, 3.0, 4.0]))

        assert {key: stats[key] for key in [
            "n_samples", "min", "max", "mean", "variance", "std", "median", "q1", "q3"
        ]} == {
            "n_samples": 4,
            "min": 1.0,
            "max": 4.0,
            "mean": 2.5,
            "variance": 1.25,
            "std": float(np.std([1.0, 2.0, 3.0, 4.0])),
            "median": 2.5,
            "q1": 1.75,
            "q3": 3.25,
        }
        assert stats["negative_ratio"] == 0.0
        assert stats["positive_ratio"] == 1.0
        assert stats["near_zero_ratio"] == 0.0
        assert stats["near_zero_threshold"] == 1e-8
        assert (stats["n_negative"], stats["n_zero"], stats["n_positive"], stats["n_near_zero"]) == (0, 0, 4, 0)
        assert (stats["n_pos_inf"], stats["n_neg_inf"], stats["n_nan"]) == (0, 0, 0)
        assert [item["threshold"] for item in stats["near_zero_fractions"]] == [1e-8, 1e-6, 1e-4]
        assert sum(item["count"] for item in stats["distribution"]["bins"]) == 4

    def test_execute_returns_sign_ratios_and_distribution_without_correlations(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = StatisticsTool(data={"x": x, "y": 2 * x}).execute(
            n_bins=2,
            near_zero_threshold=0.01,
        )

        assert "correlations" not in result
        assert result["statistics"]["x"]["negative_ratio"] == 0.25
        assert result["statistics"]["x"]["zero_ratio"] == 0.25
        assert result["statistics"]["x"]["near_zero_ratio"] == 0.25
        assert len(result["statistics"]["x"]["distribution"]["bins"]) == 2

    def test_get_stats_flattens_multidimensional_arrays(self):
        tool = StatisticsTool(data={})
        stats = tool.get_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert stats["n_samples"] == 4
        assert stats["mean"] == 2.5

    def test_get_stats_raises_for_empty_array(self):
        tool = StatisticsTool(data={})

        try:
            tool.get_stats(np.array([]))
        except ValueError as exc:
            assert "zero-size array" in str(exc)
        else:
            raise AssertionError("Expected ValueError for empty array")


class TestStatisticsToolFormatting:
    def test_format_result_dict_formats_each_variable(self):
        tool = StatisticsTool(data={})
        stat = tool.get_stats(np.array([-1.0, 0.0, 1.0, 4.0]), n_bins=2, near_zero_threshold=0.01)
        formatted = StatisticsTool.format_result_dict({"statistics": {"x1": stat}, "exceptions": []})

        assert "x1 (finite samples=4/4; finite ratio=100%):" in formatted
        assert "  Fractions:" in formatted
        assert "Inf=0/4 (0%)" in formatted
        assert "Negative inf=0/4 (0%)" in formatted
        assert "NaN=0/4 (0%)" in formatted
        assert "Negative=1/4 (25.0%);" in formatted
        assert "Zero=1/4 (25.0%);" in formatted
        assert "Positive=2/4 (50.0%);" in formatted
        assert "Statistics (computed on 4 finite values):" in formatted
        assert "Range=[-1.00, 4.00]" in formatted
        assert "Mean=1.00;" in formatted and "Median=0.500;" in formatted
        assert "Variance=3.50;" in formatted and "Std=1.87 (ddof=0);" in formatted
        assert "Q1 (25%)=-0.250;" in formatted and "Q3 (75%)=1.75;" in formatted
        assert "fraction(|x1| <= 1.00e-02)=1/4 (25.0%);" in formatted
        assert "fraction(|x1| <= 1.00e-06)=1/4 (25.0%);" in formatted
        assert "fraction(|x1| <= 1.00e-04)=1/4 (25.0%);" in formatted
        assert "Equal-width histogram (n_bins=2; finite values only" in formatted
        assert "[-1.00, 1.50) | 3 | 75.0%" in formatted

    def test_format_result_dict_formats_multiple_variables_in_order(self):
        tool = StatisticsTool(data={})
        result = {"statistics": {
            "x1": tool.get_stats(np.array([1.0]), n_bins=1),
            "y": tool.get_stats(np.array([2.0]), n_bins=1),
        }, "exceptions": []}

        formatted = StatisticsTool.format_result_dict(result)

        assert formatted.index("x1 (finite samples=") < formatted.index("y (finite samples=")

    def test_format_result_dict_includes_every_get_stats_field(self):
        tool = StatisticsTool(data={})
        stat = tool.get_stats(np.array([1.0, 2.0, np.nan]), n_bins=2)
        formatted = tool.format_result_dict({"statistics": {"x": stat}, "exceptions": []})

        assert "x (finite samples=2/3; finite ratio=66.7%):" in formatted
        assert "NaN=1/3 (33.3%)" in formatted
        assert "| 1 | 50.0%" in formatted
        assert "(Range | samples | fraction of finite samples)" in formatted
        assert "[1.00, 1.50)" in formatted and "[1.50, 2.00]" in formatted

    def test_nonfinite_and_sign_fractions_use_total_sample_denominator(self):
        x = np.array([np.inf, -np.inf, np.nan, -1.0, 0.0, 1.0])
        result = StatisticsTool(data={"x": x})(variables=["x"])
        assert result.ok
        stat = result.result["statistics"]["x"]
        assert (stat["n_pos_inf"], stat["n_neg_inf"], stat["n_nan"]) == (1, 1, 1)
        assert "Inf=1/6 (16.7%)" in result.result_str
        assert "Negative inf=1/6 (16.7%)" in result.result_str
        assert "NaN=1/6 (16.7%)" in result.result_str
        assert "Negative=1/6 (16.7%);" in result.result_str
        assert "Zero=1/6 (16.7%);" in result.result_str
        assert "Positive=1/6 (16.7%);" in result.result_str
        assert "fraction(|x| <= 1.00e-08)=1/3 (33.3%);" in result.result_str
