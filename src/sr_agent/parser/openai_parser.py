# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..api.core import ToolCall
from ..tools import BaseTool, ToolCallResult

_logger = getLogger(f'sr_agent.{__name__}')


@BaseParser.register('openai')
class OpenAIParser(BaseParser):
    """基于 OpenAI 格式的工具调用解析器。"""

    def format_tools(self) -> str:
        raise NotImplementedError("OpenAIParser.format_tools() is not implemented yet.")

    def parse_response(self, response: str) -> List[ToolCall]:
        raise NotImplementedError("OpenAIParser.parse_response() is not implemented yet.")

    def format_tool_result_messages(
        self,
        tool_calls: List[ToolCall],
        results: List[ToolCallResult | None],
    ) -> List[Dict[str, Any]]:
        messages = []
        for tool_call, result in zip(tool_calls, results):
            result_str = result.result_str if isinstance(result, ToolCallResult) else str(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": result_str,
            })
        return messages
