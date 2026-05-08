# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import BaseParser


@dataclass
class ToolCall:
    """Normalized tool call emitted by LLM APIs and parsers."""

    name: str
    params: dict
    id: str | None = None
    raw: Any = None
    raw_str: str | None = None


class LLMResult: # 这个类已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
    """
    Wrapper for LLM generator that captures the return value.

    Example:
        >>> api = OpenAIAPI(model='gpt-4o-mini')
        >>> result = api("Hello", n=3)  # Returns LLMResult
        >>> for content, tool_call in result:
        ...     print(content, tool_call)  # Stream generated content and tool calls
        >>> print(result.usage)     # Access via property
        >>> print(result.contents)  # List of generated contents
    """

    def __init__(self, gen: Iterator, parser: BaseParser = None):
        self.gen = gen
        self.parser = parser
        self.yielded = []
        self.returned: dict = {}

    def __iter__(self):
        """Iterate over the generated contents."""
        try:
            while True:
                result = next(self.gen)
                self.yielded.append(result)
                yield result['content'], result['tool_call'], result['message']
        except StopIteration as e:
            self.returned = e.value

    @property
    def usage(self) -> dict:
        """Token & Price usage statistics."""
        return self.returned['usage']

    @property
    def return_value(self) -> dict:
        """Alias for the generator return value."""
        return self.returned

    @property
    def contents(self) -> dict:
        """Raw API contents."""
        return self.returned['contents']

    @property
    def tool_calls(self) -> list:
        """Tool calls returned by the provider."""
        return self.returned['tool_calls']
