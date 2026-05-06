# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import os
import json
import logging
from copy import deepcopy
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Literal, Sequence, Tuple, Type
from ..tools import BaseTool
from ..parser import BaseParser
from .core import LLMResult, ToolCall
from .tool_call_mixin import ToolCallMixin

_logger = logging.getLogger(__name__)
ToolParserName = Literal["text", "json", "xml", "openai"]
ToolList = List[BaseTool]


class LLMAPI(ToolCallMixin):
    supported_models: list[str] = []

    @classmethod
    def create( # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        cls,
        llm_provider: str,
        llm_model: str,
        tool_list: ToolList = None,
        tool_parser: ToolParserName = "text",
    ) -> "LLMAPI":
        kwargs = {
            "model": llm_model,
            "tool_list": tool_list,
            "tool_parser_name": tool_parser,
        }
        if llm_provider:
            if llm_provider.lower() == "siliconflow":
                from .siliconflow_api import SiliconFlowAPI
                return SiliconFlowAPI(**kwargs)
            if llm_provider.lower() == "openrouter":
                from .openrouter_api import OpenRouterAPI
                return OpenRouterAPI(**kwargs)
            if llm_provider.lower() == "openai":
                from .openai_api import OpenAIAPI
                return OpenAIAPI(**kwargs)
            if llm_provider.lower() == "gemini":
                from .gemini_api import GeminiAPI
                return GeminiAPI(**kwargs)
            if llm_provider.lower() == "deepseek":
                from .deepseek_api import DeepSeekAPI
                return DeepSeekAPI(**kwargs)
            if llm_provider.lower() == "manual":
                from .manual_api import ManualAPI
                return ManualAPI(**kwargs)
            raise ValueError(f"Unsupported provider: {llm_provider}.")

        if llm_model:
            from .manual_api import ManualAPI
            if llm_model in ManualAPI.supported_models:
                return ManualAPI(**kwargs)
            from .deepseek_api import DeepSeekAPI
            if llm_model in DeepSeekAPI.supported_models:
                return DeepSeekAPI(**kwargs)
            from .openai_api import OpenAIAPI
            if llm_model in OpenAIAPI.supported_models:
                return OpenAIAPI(**kwargs)
            from .gemini_api import GeminiAPI
            if llm_model in GeminiAPI.supported_models:
                return GeminiAPI(**kwargs)
            from .siliconflow_api import SiliconFlowAPI
            if llm_model in SiliconFlowAPI.supported_models:
                return SiliconFlowAPI(**kwargs)
            from .openrouter_api import OpenRouterAPI
            if llm_model in OpenRouterAPI.supported_models:
                return OpenRouterAPI(**kwargs)
            raise ValueError(f"Unsupported model: {llm_model}.")

        _logger.warning("No llm_provider or llm_model specified, returning base LLMAPI.")
        return LLMAPI(**kwargs)

    def __init__( # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        self,
        model: str = None,
        tool_list: ToolList = None,
        tool_parser_name: ToolParserName = "text",
        parser: ToolParserName | None = None,
    ):
        self.model = model
        if parser is not None:
            tool_parser_name = parser
        self.tool_list = tool_list
        self.tool_parser_name = tool_parser_name
        self.tool_parser = self.build_parser(tool_parser_name)

    def __call__(self, messages: List | str, **kwargs) -> LLMResult:  # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        """Request the LLM and return an LLMResult wrapper.

        This is the main entry point for requesting the LLM API.
        It returns an LLMResult object that:
        - Can be iterated to get generated content (streaming)
        - Provides property access to usage, messages, response, etc.

        Args:
            messages (List | str): The input messages or prompt string.

        Returns:
            LLMResult: A wrapper that yields content and provides property access to results.

        Example:
            >>> api = OpenAIAPI(model='gpt-4o-mini')
            >>> result = api("Hello")
            >>> for content in result:
            ...     print(content)
            >>> print(result.usage)      # Token and money usage
            >>> print(result.messages)   # List of all messages in the conversation
            >>> print(result.contents)   # List of generated contents
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # if not self.tool_list:
        #     pass # No tools, just request with messages
        # elif self.tool_parser_name == "openai":
        #     kwargs['tools'] = self.tool_metadata # Pass tool metadata in kwargs for OpenAI's function calling
        # else:
        #     messages = self.add_tool_description(messages) # Add tool description to messages for non-native parsing

        generator = self._request(messages, **kwargs)
        return LLMResult(generator, self.tool_parser)

    @abstractmethod
    def _request(self, messages: List[Dict[str, str]], **kwargs) -> Generator[Tuple[str, List[ToolCall] | None], None, Dict[str, Any]]:  # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        """Send the request to the LLM API and yield (generated content, tool calls), with the final return value being the full API response dict and other metadata."""
        pass

    def build_parser(self, parser: ToolParserName) -> BaseParser | None:  # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        if not self.tool_list:
            return None # No tools available, so no parser needed
        elif parser == "openai":
            return None # OpenAI's function calling does not require a separate parser
        elif parser in {"text", "json", "xml"}:
            return BaseParser.create(parser, tool_list=self.tool_list)
        else:
            raise ValueError(f"Unsupported tool parser: {parser}")

    def setup_proxy(self):
        """Setup HTTP/HTTPS proxy from environment variables if specified."""
        if (
            (proxy := os.environ.get("MY_PROXY")) or 
            (proxy := os.environ.get("my_proxy"))
        ):
            os.environ["http_proxy"] = proxy
            os.environ["HTTP_PROXY"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
