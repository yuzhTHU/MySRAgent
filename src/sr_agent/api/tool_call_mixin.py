# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import Any, Dict, List, TYPE_CHECKING
from .core import ToolCall
if TYPE_CHECKING:
    from .llm_api import ToolParserName, ToolList

_logger = logging.getLogger(__name__)


class ToolCallMixin:
    """Shared helpers for normalizing native and text-parsed tool calls."""
    tool_list: ToolList
    tool_parser: ToolParserName

    @cached_property
    def tool_description_text(self) -> str: # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        return (
            "Use the following tools when a tool call is needed. "
            "Return tool calls in the specified format.\n\n"
            f"{self.tool_parser.format_tools()}"
        )

    @cached_property
    def tool_description_json(self) -> List[Dict]: # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters,
                },
            }
            for tool in self.tool_list
        ]

    def add_tool_description(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        """ 用 self.tool_parser.format_tools() 生成工具描述，并将其添加到 messages 中，供 LLM 参考。 """
        if (role := messages[0]["role"]) not in {"system", "developer"}:
            return [{"role": "system", "content": self.tool_description_text}] + messages
        elif self.tool_description_text not in (content := messages[0]["content"]):
            return [{'role': role, 'content': f"{content}\n\n{self.tool_description_text}"}] + messages[1:]
        else:
            return messages

    def attach_tool_calls(self, result: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        response_message = self.extract_response_message(result)
        raw_response = result.get("response", result.get("responses"))
        assistant_message = self.extract_assistant_message(raw_response)

        if self.tool_parser_name == "openai":
            tool_calls = self.normalize_native_tool_calls(
                self.extract_native_tool_calls(raw_response, assistant_message)
            )
        elif self.tool_parser is not None and response_message:
            tool_calls = self.tool_parser.parse_response(response_message)
        else:
            tool_calls = []

        result["response_message"] = response_message
        result["tool_calls"] = tool_calls
        if assistant_message:
            result["assistant_message"] = assistant_message
        return result

    def extract_response_message(self, result: Dict[str, Any]) -> str:
        contents = result.get("contents")
        if isinstance(contents, list) and contents:
            return contents[0] or ""
        if isinstance(contents, str):
            return contents

        assistant_message = self.extract_assistant_message(result.get("response"))
        if assistant_message:
            return assistant_message.get("content") or ""
        return ""

    def extract_assistant_message(self, response: Any) -> Dict[str, Any] | None:
        if response is None:
            return None
        if isinstance(response, list):
            if not response:
                return None
            response = response[0]
        if not isinstance(response, dict):
            return None

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                return message

        if response.get("role") == "assistant":
            return response

        nested = response.get("response")
        if nested is not None and nested is not response:
            return self.extract_assistant_message(nested)

        return None

    def extract_native_tool_calls(
        self,
        response: Any,
        assistant_message: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if assistant_message:
            tool_calls = assistant_message.get("tool_calls") or []
            if isinstance(tool_calls, list):
                return tool_calls

        if isinstance(response, list):
            calls = []
            for item in response:
                calls.extend(self.extract_native_tool_calls(item))
            return calls

        if not isinstance(response, dict):
            return []

        output = response.get("output")
        if isinstance(output, list):
            return [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}
            ]

        if isinstance(response.get("functionCall"), dict):
            return [response["functionCall"]]
        if isinstance(response.get("function_call"), dict):
            return [response["function_call"]]

        nested = response.get("response")
        if nested is not None and nested is not response:
            return self.extract_native_tool_calls(nested)

        return []

    def parse_native_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name")
        raw_arguments = (
            function.get("arguments")
            if "arguments" in function
            else tool_call.get("arguments", tool_call.get("arguments_json", tool_call.get("args", {})))
        )
        return ToolCall(
            name=name,
            params=self.parse_tool_arguments(raw_arguments),
            id=tool_call.get("id") or tool_call.get("call_id"),
            raw=tool_call,
        )

    def normalize_native_tool_calls(self, tool_calls: List[Any] | None) -> List[ToolCall]:
        """Normalize provider-native tool call payloads for the agent loop."""
        normalized = []
        for tool_call in tool_calls or []:
            if not isinstance(tool_call, dict):
                if hasattr(tool_call, "to_dict"):
                    tool_call = tool_call.to_dict()
                elif hasattr(tool_call, "model_dump"):
                    tool_call = tool_call.model_dump()
                else:
                    continue
            normalized.append(self.parse_native_tool_call(tool_call))
        return normalized

    def parse_tool_arguments(self, raw_arguments: Any) -> Dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            if not raw_arguments.strip():
                return {}
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                _logger.warning(f"Failed to parse tool arguments as JSON: {raw_arguments}")
                return {"raw_arguments": raw_arguments}
            return parsed if isinstance(parsed, dict) else {"arguments": parsed}
        return {"arguments": raw_arguments}
