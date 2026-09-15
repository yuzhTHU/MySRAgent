"""BaseTool 的单元测试。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import nd2py as nd
import numpy as np
import pytest

from sr_agent.tools.base_tool import BaseTool, ToolCallResult, ToolRunAbort, ToolMetadata


@BaseTool.register("unit_sample_tool")
class UnitSampleTool(BaseTool):
    metadata = ToolMetadata(name="unit_sample_tool")

    def execute(
        self,
        required_text: str,
        count: int = 1,
        ratio: float = 1.5,
        enabled: bool = True,
        names: List[str] = None,
        maybe: Optional[str] = None,
        pair: Tuple[str, int] = ("x", 1),
        mode: Literal["fast", "slow"] = "fast",
        payload: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Run a sample tool.

        Args:
            required_text: Required text input.
            count: Number of repeats.
            ratio: Floating point ratio.
            enabled: Whether the tool is enabled.
            names: Optional list of names.
            maybe: Optional string.
            pair: String and integer pair.
            mode: Execution mode.
            payload: Arbitrary object payload.

        Returns:
            Result dictionary.
        """
        return {
            "required_text": required_text,
            "count": count,
            "context": self.context,
        }


@BaseTool.register("unit_manual_schema_tool")
class UnitManualSchemaTool(BaseTool):
    metadata = ToolMetadata(
        name="unit_manual_schema_tool",
        description="Manual schema description.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def execute(self, value: str) -> Dict[str, Any]:
        """This docstring should not override manual metadata."""
        return {"value": value}


@BaseTool.register("unit_error_tool")
class UnitErrorTool(BaseTool):
    metadata = ToolMetadata(name="unit_error_tool")

    def execute(self) -> Dict[str, Any]:
        """Raise a controlled error."""
        raise RuntimeError("boom")


@BaseTool.register("unit_abort_tool")
class UnitAbortTool(BaseTool):
    metadata = ToolMetadata(name="unit_abort_tool")

    def execute(self) -> Dict[str, Any]:
        """Raise an abort error."""
        raise ToolRunAbort("stop now")


class TestToolMetadata:
    def test_metadata_defaults(self):
        metadata = ToolMetadata(name="demo")

        assert metadata.name == "demo"
        assert metadata.description is None
        assert metadata.parameters is None


class TestToolCallResult:
    def test_tool_call_result_fields(self):
        result = ToolCallResult(
            ok=True,
            result={"answer": 42},
            result_str="answer=42",
            meta_data={"tool": "demo"},
        )

        assert result.ok is True
        assert result.result == {"answer": 42}
        assert result.result_str == "answer=42"
        assert result.meta_data == {"tool": "demo"}


class TestBaseToolMetadataInference:
    def test_infer_tool_description_uses_execute_docstring_before_args(self):
        assert UnitSampleTool.metadata.description == "Run a sample tool."

    def test_manual_metadata_is_not_overwritten(self):
        assert UnitManualSchemaTool.metadata.description == "Manual schema description."
        assert UnitManualSchemaTool.metadata.parameters == {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    def test_parse_args_docstring(self):
        descriptions = BaseTool.parse_args_docstring(UnitSampleTool.execute)

        assert descriptions["required_text"] == "Required text input."
        assert descriptions["names"] == "Optional list of names."
        assert descriptions["mode"] == "Execution mode."

    def test_parse_args_docstring_fills_missing_descriptions(self):
        def func(self, value: str):
            """No Args section."""

        assert BaseTool.parse_args_docstring(func) == {
            "value": "(no description provided)",
        }

    @pytest.mark.parametrize(
        ("annotation", "schema"),
        [
            (str, {"type": "string"}),
            (int, {"type": "integer"}),
            (float, {"type": "number"}),
            (bool, {"type": "boolean"}),
            (list, {"type": "array"}),
            (dict, {"type": "object"}),
            (Any, {}),
            (List[str], {"type": "array", "items": {"type": "string"}}),
            (
                Optional[str],
                {"anyOf": [{"type": "string"}, {"type": "null"}]},
            ),
            (
                Tuple[str, str],
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ),
            (
                Literal["fast", "slow"],
                {"enum": ["fast", "slow"], "type": "string"},
            ),
        ],
    )
    def test_parse_args_typehints(self, annotation, schema):
        assert BaseTool.parse_args_typehints(annotation) == schema

    def test_parse_json_type(self):
        assert BaseTool.parse_json_type(bool) == "boolean"
        assert BaseTool.parse_json_type(int) == "integer"
        assert BaseTool.parse_json_type(float) == "number"
        assert BaseTool.parse_json_type(str) == "string"
        assert BaseTool.parse_json_type(list) == "array"
        assert BaseTool.parse_json_type(dict) == "object"
        assert BaseTool.parse_json_type(object) == ""

    def test_infer_tool_parameters(self):
        schema = UnitSampleTool.metadata.parameters

        assert schema["type"] == "object"
        assert schema["required"] == ["required_text"]
        properties = schema["properties"]
        assert properties["required_text"] == {
            "type": "string",
            "description": "Required text input.",
        }
        assert properties["count"] == {
            "type": "integer",
            "description": "Number of repeats.",
            "default": 1,
        }
        assert properties["ratio"]["type"] == "number"
        assert properties["enabled"]["type"] == "boolean"
        assert properties["names"]["items"] == {"type": "string"}
        assert properties["maybe"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
        # assert properties["pair"]["items"] == [
        #     {"type": "string"},
        #     {"type": "integer"},
        # ]
        assert properties["mode"]["enum"] == ["fast", "slow"]
        assert properties["payload"]["type"] == "object"


class TestBaseToolExportAndCall:
    def test_to_dict_exports_openrouter_tool_schema(self):
        tool_dict = UnitSampleTool.to_dict()

        assert tool_dict["type"] == "function"
        assert tool_dict["function"]["name"] == "unit_sample_tool"
        assert tool_dict["function"]["description"] == "Run a sample tool."
        assert tool_dict["function"]["parameters"] == UnitSampleTool.metadata.parameters

    def test_to_tool_list_returns_all_registered_tools_when_unfiltered(self):
        tools = BaseTool.to_tool_list()
        names = {tool["function"]["name"] for tool in tools}

        assert "unit_sample_tool" in names
        assert "unit_manual_schema_tool" in names

    def test_to_tool_list_filters_by_tool_name(self):
        tools = BaseTool.to_tool_list(["unit_sample_tool"])

        assert [tool["function"]["name"] for tool in tools] == ["unit_sample_tool"]

    def test_default_format_result_dict(self):
        assert BaseTool.format_result_dict({"value": 1}) == "{'value': 1}"

    def test_call_returns_tool_call_result_on_success(self):
        tool = UnitSampleTool(session_id="abc")
        result = tool(required_text="hello", count=3)

        assert isinstance(result, ToolCallResult)
        assert result.ok is True
        assert result.result == {
            "required_text": "hello",
            "count": 3,
            "context": {"session_id": "abc"},
        }
        assert result.result_str == str(result.result)
        assert result.meta_data["tool"] == "unit_sample_tool"
        assert result.meta_data["execution_time"] >= 0
        assert "timestamp" in result.meta_data

    def test_call_returns_tool_call_result_on_error(self):
        result = UnitErrorTool()()

        assert result.ok is False
        assert "error" in result.result
        assert "RuntimeError" in result.result_str
        assert "boom" in result.result_str
        assert "\x1b" not in result.result_str
        assert "Traceback" not in result.result_str
        assert result.result_str == (
            "Error executing unit_error_tool:\n"
            "    RuntimeError: boom"
        )
        assert result.meta_data["tool"] == "unit_error_tool"

    def test_call_removes_ansi_sequences_before_truncating(self, monkeypatch):
        monkeypatch.setattr(
            UnitSampleTool,
            "format_result_dict",
            classmethod(
                lambda cls, result: (
                    "\x1b[31mred\x1b[0m "
                    "\x1b]8;;https://example.com\x07link\x1b]8;;\x07"
                )
            ),
        )
        monkeypatch.setattr(UnitSampleTool, "MAX_RESULT_STR_LENGTH", 8)

        result = UnitSampleTool()(required_text="hello")

        assert result.result_str == "red link"
        assert "\x1b" not in result.result_str

    def test_call_does_not_catch_tool_run_abort(self):
        with pytest.raises(ToolRunAbort, match="stop now"):
            UnitAbortTool()()

    def test_result_string_is_truncated_without_mutating_raw_result(self, monkeypatch):
        monkeypatch.setattr(UnitSampleTool, "MAX_RESULT_STR_LENGTH", 80)
        payload = "x" * 200

        result = UnitSampleTool()(required_text=payload)

        assert result.result["required_text"] == payload
        assert len(result.result_str) <= 80
        assert "characters ignored" in result.result_str

    def test_formula_normalization_rejects_oversized_input(self):
        with pytest.raises(ValueError, match="Formula is too long"):
            BaseTool.normalize_formula("x" * (BaseTool.MAX_FORMULA_LENGTH + 1))


class TestBaseToolEvaluate:
    def test_formula_constants_are_serialized_with_eight_significant_digits(self):
        x = np.arange(1.0, 5.0)
        tool = UnitSampleTool(data={"x": x, "y": 2 * x}, target="y")

        result = tool.evaluate(
            f=nd.parse("1.0582314356281874*x + 1.0029397100063842"),
            y=nd.parse("y"),
            show_diagnostics=False,
        )

        assert result["formula"] == "1.0582314 * x + 1.0029397"

    def test_evaluates_required_symbols_and_computes_formula_complexity(self):
        tool = UnitSampleTool(
            data={"x": np.arange(1.0, 5.0), "y": 2 * np.arange(1.0, 5.0)},
            target="y",
        )
        f = nd.parse("2*x")
        y = nd.parse("y")

        result = tool.evaluate(f=f, y=y)
        train_result = result["data_split_results"]["train"]
        metrics = train_result["metrics"]

        assert metrics["rmse"] == 0.0
        assert metrics["complexity"] == len(f)
        assert metrics["aic"] == float("-inf")
        assert metrics["bic"] == float("-inf")
        assert result["formula"] == f.to_str()
        assert result["is_candidate"] is True
        assert train_result["diagnostics"]

    def test_calculate_metrics_reuses_external_predictions(self):
        f = nd.parse("missing_prediction")
        metrics = UnitSampleTool.calculate_metrics(
            f, np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
        assert metrics["mse"] == 0.0
        assert metrics["complexity"] == len(f)

    def test_calculate_metrics_broadcasts_prediction_and_target_symmetrically(self):
        f = nd.parse("prediction")
        assert UnitSampleTool.calculate_metrics(f, np.array([2.0, 2.0]), np.array(2.0))["mse"] == 0.0
        assert UnitSampleTool.calculate_metrics(f, np.array(2.0), np.array([2.0, 2.0]))["mse"] == 0.0

    def test_incompatible_external_shapes_raise_clear_error(self):
        with pytest.raises(ValueError, match="cannot be broadcast"):
            UnitSampleTool.calculate_metrics(nd.parse("prediction"), np.ones(2), np.ones(3))

    def test_aic_and_bic_use_number_of_fitted_constants(self):
        x = np.arange(1.0, 21.0)
        y_values = 2 * x + np.linspace(-0.2, 0.2, len(x))
        tool = UnitSampleTool(data={"x": x, "y": y_values}, target="y")
        f = nd.parse("2*x")
        y = nd.parse("y")

        metrics = tool.evaluate(f=f, y=y, show_diagnostics=False)["data_split_results"]["train"]["metrics"]
        residuals = f.eval(tool.context["data"]) - y_values
        ss_res = float(np.sum(residuals**2))
        expected_log_likelihood = -len(x) / 2 * (
            np.log(2 * np.pi) + np.log(ss_res / len(x)) + 1
        )

        n_parameters = 1
        assert metrics["aic"] == pytest.approx(2 * n_parameters - 2 * expected_log_likelihood)
        assert metrics["bic"] == pytest.approx(n_parameters * np.log(len(x)) - 2 * expected_log_likelihood)

    def test_requires_nd2py_symbols(self):
        tool = UnitSampleTool(data={})
        with pytest.raises(TypeError, match="nd2py.Symbol"):
            tool.evaluate(
                f="x",
                y=nd.parse("y"),
            )

    def test_formatted_evaluation_uses_equation_and_conditional_ineligibility_notes(self):
        x = np.arange(1.0, 6.0)
        tool = UnitSampleTool(data={"x": x, "y": 2 * x, "z": x}, target="y")
        eligible = tool.evaluate(f=nd.parse("2*x"), y=nd.parse("y"))
        wrong_lhs = tool.evaluate(f=nd.parse("x"), y=nd.parse("z"))
        target_leak = tool.evaluate(f=nd.parse("y + x"), y=nd.parse("y"))

        eligible_text = tool.format_evaluation_result(eligible, title="Best fitted rational formula")
        wrong_lhs_text = tool.format_evaluation_result(wrong_lhs, title="Best fitted rational formula")
        target_leak_text = tool.format_evaluation_result(target_leak, title="Best fitted rational formula")
        assert "Best fitted rational formula:\n    y = " in eligible_text
        assert "Fit quality (Train-set | Validation-set):\n    RMSE=0.00 | N/A;" in eligible_text
        assert "Error extremes (Top-10 sorted by |residual|):\n    (x | z | y | residual)" in eligible_text
        assert "not eligible for submission" not in eligible_text
        assert "the left-hand side of the equation is not y" in wrong_lhs_text
        assert "the right-hand side of the equation depends on y" in target_leak_text

    def test_formatted_evaluation_includes_independent_validation_when_supplied(self):
        x = np.arange(1.0, 6.0)
        tool = UnitSampleTool(
            data={"x": x, "y": 2 * x},
            evaluation_data={"x": x + 5, "y": 2 * (x + 5)},
            target="y",
        )
        evaluation = tool.evaluate(f=nd.parse("2*x"), y=nd.parse("y"))
        text = tool.format_evaluation_result(evaluation)
        assert "Fit quality (Train-set | Validation-set):\n    RMSE=0.00 | 0.00;\n    MAE=0.00 | 0.00;\n    R2=1.00 | 1.00;" in text

    def test_formatted_evaluation_uses_eight_significant_digits_for_formula_only(self):
        result = {
            "formula": "0.123456789012 * x",
            "target_expression": "y",
            "data_split_results": {
                "train": {
                    "metrics": {"rmse": 0.123456, "mae": 1.4, "r2": -107.015, "complexity": 16},
                    "diagnostics": {
                        "worst_samples": [
                            {
                                "row": {"x": 5.348730564},
                                "y_true": 0.2041338086,
                                "y_pred": -36.56052103,
                            },
                            {
                                "row": {"x": 1.4},
                                "y_true": 1,
                                "y_pred": 2,
                            },
                        ],
                        "strongest_residual_correlations": [{
                            "variable": "x", "pearson": -0.003459, "spearman": 0.01807,
                        }],
                    },
                },
            },
        }
        text = UnitSampleTool.format_evaluation_result(result)
        assert result["formula"] == "0.123456789012 * x"
        assert "y = 0.12345679 * x" in text
        assert "RMSE=0.123 | N/A;" in text
        assert "MAE=1.40 | N/A;" in text
        assert "R2=-107 | N/A;" in text
        assert "Formula Complexity=16.0;" in text
        assert "5.35 | 0.204 | -36.8" in text
        assert "1.40 | 1.00 | 1.00" in text
        assert "Pearson(residual, x)=-0.00346;" in text
        assert "Spearman(residual, x)=0.0181;" in text

    def test_formula_display_preserves_exact_symbolic_constants(self):
        result = {
            "formula": "1.234567891e-6*x + x**(Number(4)/Number(3)) + pi",
            "target_expression": "y",
            "data_split_results": {
                "train": {
                    "metrics": {"rmse": 0, "mae": 0, "r2": 1, "complexity": 10},
                },
            },
        }

        text = UnitSampleTool.format_evaluation_result(result)

        assert "1.2345679e-06 * x" in text
        assert "x ** (4 / 3)" in text
        assert "+ pi" in text

    def test_free_form_formula_description_is_preserved_when_not_parseable(self):
        result = {
            "formula": "piecewise model: branch A, then branch B",
            "data_split_results": {
                "train": {
                    "metrics": {"rmse": 0, "mae": 0, "r2": 1, "complexity": 10},
                },
            },
        }

        text = UnitSampleTool.format_evaluation_result(result)

        assert "LHS = piecewise model: branch A, then branch B" in text
