# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""Base class and shared tool-call support for LLM provider adapters."""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Generator, List, Literal

from ..core import APICallResult, ToolCall
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import FactoryMixin, log_exception

_logger = logging.getLogger(f"sr_harness.{__name__}")

ToolParserName = Literal["text", "json", "xml", "openai"]
ToolList = list[BaseTool | type[BaseTool]]


class BaseAPI(ABC, FactoryMixin):
    """Common request, parser, and tool-call behavior for LLM providers."""

    supported_models: list[str] = []

    def __init__(
        self,
        model: str | None = None,
        tool_list: ToolList | None = None,
        tool_parser_name: ToolParserName = "text",
        parser: ToolParserName | None = None,
    ):
        self.model = model
        if parser is not None:
            tool_parser_name = parser
        self.tool_list = tool_list
        self.tool_parser_name = tool_parser_name
        self.tool_parser = self.build_parser(tool_parser_name)

    def __call__(self, messages: List | str, **kwargs) -> APICallResult:
        """Request the provider and wrap its result generator."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        generator = self._request(messages, **kwargs)
        return APICallResult(generator, self.tool_parser)

    @abstractmethod
    def _request(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        """Yield provider responses and return aggregate response metadata."""
        raise NotImplementedError

    def build_parser(self, parser: ToolParserName) -> BaseParser | None:
        if not self.tool_list or parser == "openai":
            return None
        if parser in {"text", "json", "xml"}:
            return BaseParser.create(parser, tool_list=self.tool_list)
        raise ValueError(f"Unsupported tool parser: {parser}")

    @cached_property
    def tool_description_text(self) -> str:
        """Format available tools for a model using a text-based parser."""
        return (
            "Use the following tools when a tool call is needed. "
            "Return tool calls in the specified format.\n\n"
            f"{self.tool_parser.format_tools()}"
        )

    @cached_property
    def tool_description_json(self) -> List[Dict]:
        """Build OpenAI-compatible native function descriptions."""
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

    def add_tool_description(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Add text-formatted tool instructions to the leading system message."""
        if (role := messages[0]["role"]) not in {"system", "developer"}:
            return [{"role": "system", "content": self.tool_description_text}] + messages
        if self.tool_description_text not in (content := messages[0]["content"]):
            return [
                {"role": role, "content": f"{content}\n\n{self.tool_description_text}"},
                *messages[1:],
            ]
        return messages

    def normalize_openai_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """Normalize provider-native function calls into internal ToolCall objects."""
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
            try:
                normalized.append(self._parse_native_tool_call(tool_call))
            except json.JSONDecodeError as error:
                _logger.warning(
                    f"Skip tool call {tool_call!r} since it cannot be parsed as JSON: "
                    f"{log_exception(error)}."
                )
        return normalized

    def _parse_native_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or None
        params = (
            function.get("arguments")
            or tool_call.get("arguments")
            or tool_call.get("arguments_json")
            or tool_call.get("args")
            or {}
        )
        if isinstance(params, str):
            params = json.loads(params) if params.strip() else {}
        return ToolCall(
            name=name,
            params=params,
            id=tool_call.get("id") or tool_call.get("call_id"),
            raw=tool_call,
        )

    def setup_proxy(self) -> None:
        """Configure HTTP/HTTPS proxy variables from MY_PROXY when provided."""
        if (proxy := os.environ.get("MY_PROXY")) or (proxy := os.environ.get("my_proxy")):
            os.environ["http_proxy"] = proxy
            os.environ["HTTP_PROXY"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
