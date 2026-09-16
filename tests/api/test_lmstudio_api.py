"""Tests for the LM Studio API adapter."""

from __future__ import annotations

from typing import Any

from sr_agent.api.core import ToolCall
from sr_agent.api.llm_api import LLMAPI
from sr_agent.api.lmstudio_api import LMStudioAPI
from sr_agent.tools import BaseTool, ToolMetadata


class DemoTool(BaseTool):
    metadata = ToolMetadata(
        name="demo_tool",
        description="Return a demo value.",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )

    def execute(self, x: int):
        return {"x": x}


class _FakeResponse:
    def __init__(self, body: dict[str, Any]):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class _FakeSession:
    instances = []
    body = {}

    def __init__(self):
        self.trust_env = True
        self.calls = []
        self.__class__.instances.append(self)

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.body)


def _consume(result):
    chunks = list(result)
    return chunks, result.return_value


def test_lmstudio_factory_and_native_tool_call(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_ENDPOINT", "http://lmstudio.test:1234/api/v1/chat")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "test-key")
    monkeypatch.setattr("sr_agent.api.lmstudio_api.requests.Session", _FakeSession)
    _FakeSession.instances = []
    _FakeSession.body = {
        "model": "qwen_qwen3-4b-instruct-2507",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "demo_tool", "arguments": '{"x": 4}'},
                }],
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }

    api = LLMAPI.create(
        "lmstudio",
        "qwen_qwen3-4b-instruct-2507",
        tool_list=[DemoTool],
        tool_parser="openai",
    )
    chunks, returned = _consume(api("use the tool", temperature=0))

    assert isinstance(api, LMStudioAPI)
    session = _FakeSession.instances[0]
    assert session.trust_env is False
    url, request = session.calls[0]
    assert url == "http://lmstudio.test:1234/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["tools"][0]["function"]["name"] == "demo_tool"
    expected = ToolCall(
        "demo_tool",
        {"x": 4},
        id="call_1",
        raw=_FakeSession.body["choices"][0]["message"]["tool_calls"][0],
    )
    assert chunks[0][1] == [expected]
    assert returned["tool_calls"] == [expected]
    assert returned["usage"]["token"] == {"prompt": 10, "answer": 6}
    assert returned["usage"]["price"] == {"total": 0.0}


def test_lmstudio_plain_chat_and_reasoning_usage(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_ENDPOINT", "http://lmstudio.test:1234/v1/chat/completions")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "test-key")
    monkeypatch.setattr("sr_agent.api.lmstudio_api.requests.Session", _FakeSession)
    _FakeSession.instances = []
    _FakeSession.body = {
        "choices": [{"message": {"role": "assistant", "content": "LMSTUDIO_OK"}}],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 7,
            "total_tokens": 23,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }

    api = LMStudioAPI()
    chunks, returned = _consume(api("ping", max_tokens=32))

    assert chunks[0][0] == "LMSTUDIO_OK"
    assert returned["contents"] == ["LMSTUDIO_OK"]
    assert returned["usage"]["token"] == {"prompt": 16, "answer": 5, "reason": 2}


def test_lmstudio_rejects_invalid_endpoint():
    import pytest

    with pytest.raises(ValueError, match="Invalid LMSTUDIO_ENDPOINT"):
        LMStudioAPI.normalize_endpoint("=not-a-url")
