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
        """ 用 self.tool_parser.format_tools() 生成工具描述文本，供 LLM 参考。 """
        return (
            "Use the following tools when a tool call is needed. "
            "Return tool calls in the specified format.\n\n"
            f"{self.tool_parser.format_tools()}"
        )

    @cached_property
    def tool_description_json(self) -> List[Dict]: # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        """ 生成 OpenAI function-calling 风格的工具描述 JSON 列表，供 LLM 参考。 """
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
        """ 将工具描述添加到 messages 中供 LLM 参考。 """
        if (role := messages[0]["role"]) not in {"system", "developer"}:
            return [{"role": "system", "content": self.tool_description_text}] + messages
        elif self.tool_description_text not in (content := messages[0]["content"]):
            return [{'role': role, 'content': f"{content}\n\n{self.tool_description_text}"}] + messages[1:]
        else:
            return messages

    def normalize_openai_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """将 OpenAI 格式的工具调用规范化为内部 ToolCall 对象"""
        normalized = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                pass
            elif hasattr(tool_call, "to_dict"):
                tool_call = tool_call.to_dict()
            elif hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump()
            else:
                raise ValueError(f"Unrecognized tool call format: {tool_call}")
            normalized.append(self._parse_native_tool_call(tool_call))
        return normalized

    def _parse_native_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or None
        params = function.get("arguments") or tool_call.get("arguments") or tool_call.get("arguments_json") or tool_call.get("args") or {}
        if isinstance(params, str):
            params = json.loads(params) if params.strip() else {}
        return ToolCall(
            name=name,
            params=params,
            id=tool_call.get("id") or tool_call.get("call_id"),
            raw=tool_call,
        )
