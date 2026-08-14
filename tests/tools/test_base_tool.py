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
        assert result.meta_data["tool"] == "unit_error_tool"

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
    def test_evaluates_required_symbols_and_computes_formula_complexity(self):
        tool = UnitSampleTool(
            data={"x": np.arange(1.0, 5.0), "y": 2 * np.arange(1.0, 5.0)},
            target="y",
        )
        f = nd.parse("2*x")
        y = nd.parse("y")

        result = tool.evaluate(f=f, y=y)
        metrics = result["metrics"]

        assert metrics["rmse"] == 0.0
        assert metrics["complexity"] == len(f)
        assert metrics["aic"] == float("-inf")
        assert metrics["bic"] == float("-inf")
        assert result["formula"] == f.to_str()
        assert result["is_candidate"] is True
        assert result["diagnostics"]

    def test_cached_arrays_avoid_symbol_evaluation(self):
        tool = UnitSampleTool(data={}, target="missing_target")
        f = nd.parse("missing_prediction")
        y = nd.parse("missing_target")

        result = tool.evaluate(
            f=f,
            y=y,
            y_pred=np.array([1.0, 2.0]),
            y_true=np.array([1.0, 2.0]),
            show_diagnostics=False,
        )

        metrics = result["metrics"]
        assert metrics["mse"] == 0.0
        assert metrics["complexity"] == len(f)
        assert "diagnostics" not in result

    def test_cached_prediction_and_target_are_broadcast_symmetrically(self):
        tool = UnitSampleTool(data={}, target="target")
        f = nd.parse("prediction")
        y = nd.parse("target")

        prediction_scalar = tool.evaluate(
            f=f, y=y, y_pred=np.array(2.0), y_true=np.array([2.0, 2.0]),
            show_diagnostics=False,
        )
        target_scalar = tool.evaluate(
            f=f, y=y, y_pred=np.array([2.0, 2.0]), y_true=np.array(2.0),
            show_diagnostics=False,
        )

        assert prediction_scalar["metrics"]["mse"] == 0.0
        assert target_scalar["metrics"]["mse"] == 0.0

    def test_incompatible_cached_shapes_raise_clear_error(self):
        tool = UnitSampleTool(data={})
        with pytest.raises(ValueError, match="cannot be broadcast"):
            tool.evaluate(
                f=nd.parse("prediction"),
                y=nd.parse("target"),
                y_pred=np.ones(2),
                y_true=np.ones(3),
                show_diagnostics=False,
            )

    def test_aic_and_bic_use_number_of_fitted_constants(self):
        x = np.arange(1.0, 21.0)
        y_values = 2 * x + np.linspace(-0.2, 0.2, len(x))
        tool = UnitSampleTool(data={"x": x, "y": y_values}, target="y")
        f = nd.parse("2*x")
        y = nd.parse("y")

        metrics = tool.evaluate(f=f, y=y, show_diagnostics=False)["metrics"]
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
                y_pred=np.array([1.0]),
                y_true=np.array([1.0]),
            )
