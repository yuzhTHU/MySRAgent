# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Template-guided JSON parsing for LLM outputs."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

import yaml


def parse_json_with_template(text: str, template: Any) -> Any:
    """Parse an LLM response and coerce it to the shape described by template.

    Template examples:
        {"score": float, "detail": [{"claim_id": str, "score": float, "detail": str}]}
        {"equation_list": [str], "detail": str}
        [{"key1": float, "key2": str}]
    """
    errors = []
    for candidate in _candidate_texts(text):
        for loader in (_loads_json, _loads_python_literal, _loads_yaml):
            try:
                return _coerce_to_template(loader(candidate), template)
            except Exception as exc:
                errors.append(exc)
    raise ValueError(f"Could not parse response with template. Last errors: {errors[-3:]}")


def _candidate_texts(text: str) -> list[str]:
    text = text.strip()
    candidates = []

    fenced_blocks = re.findall(r"```(?:json|python)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    candidates.extend(_balanced_snippets(text))
    if text:
        candidates.append(text)

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _balanced_snippets(text: str) -> list[str]:
    snippets = []
    stack = []
    start = None
    quote = None
    escaped = False

    pairs = {"{": "}", "[": "]"}
    opens = set(pairs)
    closes = set(pairs.values())

    for idx, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            continue

        if char in opens:
            if not stack:
                start = idx
            stack.append(pairs[char])
        elif char in closes and stack:
            if char != stack[-1]:
                stack.clear()
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                snippets.append(text[start : idx + 1])
                start = None

    return snippets


def _loads_json(text: str) -> Any:
    return json.loads(text)


def _loads_python_literal(text: str) -> Any:
    return ast.literal_eval(text)


def _loads_yaml(text: str) -> Any:
    return yaml.safe_load(text)


def _coerce_to_template(value: Any, template: Any) -> Any:
    if template is Any:
        return value

    if isinstance(template, Mapping):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping, got {type(value).__name__}")
        return {key: _coerce_to_template(value[key], child) for key, child in template.items()}

    if isinstance(template, list):
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        if len(template) == 0:
            return value
        if len(template) != 1:
            if len(value) != len(template):
                raise ValueError(f"Expected list of length {len(template)}, got {len(value)}")
            return [_coerce_to_template(item, child) for item, child in zip(value, template)]
        return [_coerce_to_template(item, template[0]) for item in value]

    if template is float:
        return _coerce_float(value)
    if template is int:
        return int(value)
    if template is str:
        return str(value)
    if template is bool:
        return _coerce_bool(value)
    if isinstance(template, type):
        return template(value)

    return value


def _coerce_float(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
    return float(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return bool(value)
