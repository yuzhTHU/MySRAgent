"""Unit tests for representative LLM API tool handling."""

from __future__ import annotations

from typing import Any

from sr_agent.api.openrouter_api import OpenRouterAPI
from sr_agent.api.siliconflow_api import SiliconFlowAPI
from sr_agent.api.core import ToolCall
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


class _FakeUsage:
    prompt_tokens = 7
    completion_tokens = 11


class _FakeOpenRouterMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def to_dict(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


class _FakeOpenRouterChoice:
    def __init__(self, message: _FakeOpenRouterMessage):
        self.message = message


class _FakeOpenRouterCompletion:
    def __init__(self, message_dict: dict):
        self.usage = _FakeUsage()
        content = message_dict.get("content") or ""
        tool_calls = message_dict.get("tool_calls") or []
        self.choices = [_FakeOpenRouterChoice(_FakeOpenRouterMessage(content, tool_calls))]
        self._message_dict = {"role": "assistant", **message_dict}

    def to_dict(self):
        return {
            "choices": [{"message": self._message_dict}],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
        }


class _FakeOpenRouterClient:
    payloads: list[dict[str, Any]] = []
    message: dict[str, Any] = {"content": "ok"}

    def __init__(self, *args, **kwargs):
        self.chat = self
        self.completions = self

    def create(self, **payload):
        self.payloads.append(payload)
        return _FakeOpenRouterCompletion(self.message)


class _FakeSiliconFlowResponse:
    def __init__(self, response: dict, status_code: int = 200):
        self._response = response
        self.status_code = status_code
        self.text = str(response)

    def json(self):
        return self._response


def _siliconflow_response(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", **message}}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 9,
            "total_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _consume(result):
    chunks = list(result)
    return chunks, result.return_value


def test_openrouter_native_tools_are_sent_and_tool_calls_are_extracted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("sr_agent.api.openrouter_api.OpenAI", _FakeOpenRouterClient)
    _FakeOpenRouterClient.payloads = []
    _FakeOpenRouterClient.message = {
        "content": "ready",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "demo_tool", "arguments": '{"x": 1}'},
            }
        ],
    }

    api = OpenRouterAPI(
        model="qwen/qwen3.6-flash",
        parser="openai",
        tool_list=[DemoTool],
    )
    chunks, return_value = _consume(api([{"role": "user", "content": "use the tool"}]))

    payload = _FakeOpenRouterClient.payloads[0]
    assert payload["model"] == "qwen/qwen3.6-flash"
    assert payload["tools"][0]["function"]["name"] == "demo_tool"
    assert payload["tool_choice"] == "auto"
    expected_message = {
        "role": "assistant",
        "content": "ready",
        "tool_calls": _FakeOpenRouterClient.message["tool_calls"],
    }
    assert chunks == [("ready", [ToolCall("demo_tool", {"x": 1}, id="call_1", raw=_FakeOpenRouterClient.message["tool_calls"][0])], expected_message)]
    assert return_value["response_message"] == "ready"
    assert return_value["tool_calls"] == [ToolCall("demo_tool", {"x": 1}, id="call_1", raw=_FakeOpenRouterClient.message["tool_calls"][0])]


def test_openrouter_text_parser_injects_tool_prompt_and_parses_action(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("sr_agent.api.openrouter_api.OpenAI", _FakeOpenRouterClient)
    _FakeOpenRouterClient.payloads = []
    _FakeOpenRouterClient.message = {"content": "Action: demo_tool(x=2)"}

    api = OpenRouterAPI(
        model="qwen/qwen3.6-flash",
        parser="text",
        tool_list=[DemoTool],
    )
    _chunks, return_value = _consume(api([{"role": "user", "content": "call demo"}]))

    payload = _FakeOpenRouterClient.payloads[0]
    assert "tools" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert "## Available Tools:" in payload["messages"][0]["content"]
    assert return_value["tool_calls"] == [ToolCall("demo_tool", {"x": 2}, raw_str="Action: demo_tool(x=2)")]


def test_siliconflow_qwen_text_parser_injects_tool_prompt_and_parses_action(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    payloads = []

    def fake_request(method, url, json, headers):
        payloads.append(json)
        return _FakeSiliconFlowResponse(
            _siliconflow_response({"content": "Action: demo_tool(x=3)"})
        )

    monkeypatch.setattr("sr_agent.api.siliconflow_api.requests.request", fake_request)

    api = SiliconFlowAPI(model="Qwen3-8B", parser="text", tool_list=[DemoTool])
    _chunks, return_value = _consume(api([{"role": "user", "content": "call demo"}]))

    payload = payloads[0]
    assert payload["model"] == "Qwen/Qwen3-8B"
    assert payload["enable_thinking"] is True
    assert "tools" not in payload
    assert "## Available Tools:" in payload["messages"][0]["content"]
    assert return_value["tool_calls"] == [ToolCall("demo_tool", {"x": 3}, raw_str="Action: demo_tool(x=3)")]


def test_siliconflow_qwen_native_tools_are_sent_and_tool_calls_are_extracted(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    payloads = []

    def fake_request(method, url, json, headers):
        payloads.append(json)
        return _FakeSiliconFlowResponse(
            _siliconflow_response({
                "content": "ready",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "demo_tool", "arguments": '{"x": 4}'},
                    }
                ],
            })
        )

    monkeypatch.setattr("sr_agent.api.siliconflow_api.requests.request", fake_request)

    api = SiliconFlowAPI(model="Qwen3-8B", parser="openai", tool_list=[DemoTool])
    chunks, return_value = _consume(api([{"role": "user", "content": "call demo"}]))

    payload = payloads[0]
    assert payload["tools"][0]["function"]["name"] == "demo_tool"
    assert payload["tool_choice"] == "auto"
    expected = [ToolCall("demo_tool", {"x": 4}, raw=_siliconflow_response({
        "content": "ready",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "demo_tool", "arguments": '{"x": 4}'},
            }
        ],
    })["choices"][0]["message"]["tool_calls"][0])]
    assert chunks == [("ready", expected, {
        "role": "assistant",
        "content": "ready",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "demo_tool", "arguments": '{"x": 4}'},
            }
        ],
    })]
    assert return_value["tool_calls"] == expected
