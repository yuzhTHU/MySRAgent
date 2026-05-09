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
                }
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

        assert set(result.keys()) == {"statistics", "exceptions"}
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
        assert "Variable 'x1'" in result.result_str
        assert result.meta_data["tool"] == "statistics_analysis"


class TestStatisticsToolStats:
    def test_get_stats_computes_all_metrics(self):
        tool = StatisticsTool(data={})
        stats = tool.get_stats(np.array([1.0, 2.0, 3.0, 4.0]))

        assert stats == {
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
        result = {
            "statistics": {
                "x1": {
                    "n_samples": 4,
                    "min": 1.0,
                    "max": 4.0,
                    "mean": 2.5,
                    "variance": 1.25,
                    "std": 1.11803398875,
                    "median": 2.5,
                    "q1": 1.75,
                    "q3": 3.25,
                }
            },
            "exceptions": [],
        }

        assert StatisticsTool.format_result_dict(result) == (
            "Variable 'x1': n=4, min=1.0000, max=4.0000, "
            "mean=2.5000, variance=1.2500, std=1.1180, "
            "median=2.5000, q1=1.7500, q3=3.2500\n"
        )

    def test_format_result_dict_formats_multiple_variables_in_order(self):
        result = {
            "statistics": {
                "x1": {
                    "n_samples": 1,
                    "min": 1.0,
                    "max": 1.0,
                    "mean": 1.0,
                    "variance": 0.0,
                    "std": 0.0,
                    "median": 1.0,
                    "q1": 1.0,
                    "q3": 1.0,
                },
                "y": {
                    "n_samples": 1,
                    "min": 2.0,
                    "max": 2.0,
                    "mean": 2.0,
                    "variance": 0.0,
                    "std": 0.0,
                    "median": 2.0,
                    "q1": 2.0,
                    "q3": 2.0,
                },
            },
            "exceptions": [],
        }

        formatted = StatisticsTool.format_result_dict(result)

        assert formatted.splitlines()[0].startswith("Variable 'x1':")
        assert formatted.splitlines()[1].startswith("Variable 'y':")
