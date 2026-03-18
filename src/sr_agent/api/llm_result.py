# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""Utilities for llmsr module"""
from collections.abc import Iterator
from typing import Any


class LLMResult:
    """
    Wrapper for LLM generator that captures the return value.

    This class wraps a generator and provides convenient property access
    to the result dictionary after the generator is exhausted.

    The key is using `yield from` to delegate to the inner generator,
    which automatically captures the return value via StopIteration.value.

    Example:
        >>> api = OpenAIAPI(model='gpt-4o-mini')
        >>> result = api("Hello")  # Returns LLMResult
        >>> for content in result:
        ...     print(content)
        >>> print(result.usage)     # Access via property
        >>> print(result.contents)  # List of generated contents
    """

    def __init__(self, gen: Iterator):
        self._gen = gen
        self._return_value: dict = {}
        self._consumed = False
        self._contents: list = []

    def _consume(self):
        """Consume the generator using yield from to capture return value."""
        if self._consumed:
            return

        self._consumed = True
        # Use yield from inside a wrapper generator to capture return value
        def wrapper():
            self._return_value = yield from self._gen

        # Consume the wrapper generator
        for item in wrapper():
            self._contents.append(item)

    def __iter__(self):
        """Iterate over the generated contents."""
        self._consume()
        return iter(self._contents)

    @property
    def _result(self) -> dict:
        """Ensure generator is consumed and return the result dict."""
        self._consume()
        return self._return_value

    @property
    def usage(self) -> dict:
        """Token usage statistics."""
        return self._result.get('usage', {})

    @property
    def messages(self) -> list:
        """Input messages."""
        return self._result.get('messages', [])

    @property
    def response(self) -> dict:
        """Raw API response."""
        return self._result.get('response', {})

    @property
    def responses(self) -> list:
        """Raw API responses (for n>1 generations)."""
        return self._result.get('responses', [])

    @property
    def contents(self) -> list:
        """List of generated content strings."""
        self._consume()
        return self._contents

    def __repr__(self) -> str:
        if not self._consumed:
            return "<LLMResult pending>"
        return f"<LLMResult contents={len(self._contents)}, usage={self._return_value.get('usage', {})}>"
