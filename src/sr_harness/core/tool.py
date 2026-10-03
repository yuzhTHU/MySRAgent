# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Shared data contracts for tool definitions, calls, and results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolMetadata:
    """Description and parameter schema exposed for a tool.

    ``description`` and ``parameters`` may be omitted so ``BaseTool`` can infer
    them from the implementation's signature and docstring.
    """

    name: str
    description: str | None = None
    parameters: Dict[str, Any] | None = None


@dataclass
class ToolCall:
    """Normalized tool call emitted by LLM APIs and parsers."""

    name: str
    params: dict
    id: str | None = None
    raw: Any = None
    raw_str: str | None = None


@dataclass
class ToolCallResult:
    """Structured result returned by the tool execution boundary.

    ``result`` retains the complete machine-readable value, while
    ``result_str`` is the bounded representation returned to the model.
    """

    ok: bool
    result: Dict[str, Any]
    result_str: str
    meta_data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.result.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.result[key]
