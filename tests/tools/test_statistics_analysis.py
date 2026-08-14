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
                    "description": "Absolute-value threshold used to count near-zero samples.",
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
        assert "Variable or expression: 'x1'" in result.result_str
        assert result.meta_data["tool"] == "statistics_analysis"


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

        assert "Variable or expression: 'x1'" in formatted
        assert "4 finite values out of 4 total samples" in formatted
        assert "finite value ratio: 100.0%" in formatted
        assert "minimum=-1" in formatted and "maximum=4" in formatted
        assert "mean=1" in formatted and "median=0.5" in formatted
        assert "variance=3.5" in formatted and "standard deviation=1.87083" in formatted
        assert "first quartile (25%)=-0.25" in formatted
        assert "third quartile (75%)=1.75" in formatted
        assert "negative value ratio: 25.0%" in formatted
        assert "exact-zero value ratio: 25.0%" in formatted
        assert "positive value ratio: 50.0%" in formatted
        assert "absolute value <= 0.01" in formatted
        assert "Equal-width histogram: 2 bins" in formatted
        assert "sample count=3 (75.0% of finite values)" in formatted

    def test_format_result_dict_formats_multiple_variables_in_order(self):
        tool = StatisticsTool(data={})
        result = {"statistics": {
            "x1": tool.get_stats(np.array([1.0]), n_bins=1),
            "y": tool.get_stats(np.array([2.0]), n_bins=1),
        }, "exceptions": []}

        formatted = StatisticsTool.format_result_dict(result)

        assert formatted.index("Variable or expression: 'x1'") < formatted.index("Variable or expression: 'y'")

    def test_format_result_dict_includes_every_get_stats_field(self):
        tool = StatisticsTool(data={})
        stat = tool.get_stats(np.array([1.0, 2.0, np.nan]), n_bins=2)
        formatted = tool.format_result_dict({"statistics": {"x": stat}, "exceptions": []})

        assert "2 finite values out of 3 total samples" in formatted
        assert "50.0% of finite values" in formatted
        assert "Bin 1:" in formatted and "Bin 2:" in formatted
