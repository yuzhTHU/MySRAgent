"""BaseTool 的单元测试。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import pytest

from sr_agent.tools.base_tool import BaseTool, ToolCallResult, ToolMetadata


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
                Tuple[str, int],
                {
                    "type": "array",
                    "items": [{"type": "string"}, {"type": "integer"}],
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
        assert properties["pair"]["items"] == [
            {"type": "string"},
            {"type": "integer"},
        ]
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
