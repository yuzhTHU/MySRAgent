# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import Generator, List, Dict
from .llm_api import LLMAPI
from ..utils import log_exception

_logger = logging.getLogger(f"sr_agent.{__name__}")


class OpenRouterAPI(LLMAPI):
    supported_models = [
        "qwen/qwen3.6-flash",
        "moonshotai/kimi-k2",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "openai/gpt-5.5",
        "openai/gpt-5.4-mini",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.1-flash-lite-preview",
        "~anthropic/claude-sonnet-latest",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-0813",
        "deepseek/deepseek-v4-flash",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-max-preview",
        "qwen/qwen3.6-plus",
        "z-ai/glm-5-turbo",
    ]

    def __init__(self, model='qwen/qwen3.6-plus', **kwargs):
        super().__init__(model=model, **kwargs)

    def _request(
        self,
        messages: List[Dict[str, str]],
        n=1,
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
    ) -> Generator[str, None, Dict]:
        yield from []
        load_dotenv()
        self.setup_proxy()
        api_key = os.environ["OPENROUTER_API_KEY"]
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        payload = {
            "model": self.model,
            "messages": messages,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model == "openai/gpt-oss-120b":
            # Keep benchmark inference reproducible across OpenRouter calls. The
            # default router otherwise mixes providers with different weight
            # quantizations (for example bf16 and fp8) in one experiment.
            payload["extra_body"] = {
                "provider": {
                    "order": ["AkashML"],
                    "allow_fallbacks": False,
                }
            }
        if not self.tool_list:
            pass
        elif self.tool_parser:
            payload["messages"] = self.add_tool_description(payload["messages"])
        else:
            payload["tools"] = self.tool_description_json
            payload["tool_choice"] = "auto"

        def get_tool_call(message, content):
            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            elif 'tool_calls' in message:
                tool_call = self.normalize_openai_tool_calls(message['tool_calls'])
                message['tool_calls'] = [call.raw for call in tool_call]
            else:
                tool_call = []
            return tool_call

        def get_usage(completion):
            token_usage = {}
            price_usage = {}
            if usage := completion.usage:
                token_usage["prompt"] = usage.prompt_tokens
                token_usage["answer"] = usage.completion_tokens
                total_tokens = getattr(usage, "total_tokens", usage.prompt_tokens + usage.completion_tokens)
                if (other := total_tokens - usage.prompt_tokens - usage.completion_tokens) > 0:
                    token_usage["others"] = other
                price_usage['total'] = getattr(usage, "cost", 0)
            return token_usage, price_usage

        # OpenRouter does not support `n` parameter now,
        # see https://github.com/OpenRouterTeam/openrouter-runner/issues/99
        details = []
        for idx in range(1, n + 1):
            max_retry = 3
            for attempt in range(1, max_retry + 1):
                try:
                    completion = client.chat.completions.create(**payload)
                    response_dict = completion.to_dict()
                    message = completion.choices[0].message.to_dict()
                    content = message['content'] or ""
                except Exception as e:
                    tool_call = []
                    retry_error = e
                else:
                    tool_call = get_tool_call(message, content)
                    if not (tool_call or content.strip()):
                        retry_error = ValueError(f"OpenRouterAPI({self.model}) returned empty content and no usable tool calls.")
                    else:
                        retry_error = None
                if retry_error is None:
                    token_usage, price_usage = get_usage(completion)
                    break
                elif attempt < max_retry:
                    _logger.error(f"Error requesting OpenRouterAPI({self.model}) since {log_exception(retry_error, with_traceback=False)}")
                    delay = self._retry_delay(retry_error, attempt)
                    _logger.info(f"Retrying OpenRouterAPI({self.model}) in {delay:g}s (attempt {attempt}/{max_retry}).")
                    time.sleep(delay)
                else:
                    _logger.error(f"Error requesting OpenRouterAPI({self.model}) since {log_exception(retry_error, with_traceback=False)}")
                    raise RuntimeError(f"OpenRouterAPI({self.model}) failed after {max_retry} attempts for sample {idx}/{n}.") from retry_error

            details.append({
                "content": content,
                "tool_call": tool_call,
                "token_usage": token_usage,
                "price_usage": price_usage,
                "response": response_dict,
            })
            yield {'content': content, 'tool_call': tool_call, 'message': message}

        token_usage = {}
        for detail in details:
            for key, value in detail["token_usage"].items():
                token_usage[key] = token_usage.get(key, 0) + value
        price_usage = {}
        for detail in details:
            for key, value in detail["price_usage"].items():
                price_usage[key] = price_usage.get(key, 0) + value
        return {
            "usage": {"token": token_usage, "price": price_usage},
            "contents": [detail["content"] for detail in details],
            "response_message": details[0]["content"] if details else "",
            "tool_calls": details[0]["tool_call"] if len(details) == 1 else [detail["tool_call"] for detail in details],
            "responses": [detail["response"] for detail in details],
        }

    @staticmethod
    def _retry_delay(error: Exception, attempt: int) -> float:
        """Honor OpenRouter's retry hint, falling back to bounded backoff."""
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                metadata = body.get("error", body).get("metadata", {})
                retry_headers = metadata.get("headers", {}) if isinstance(metadata, dict) else {}
                retry_after = retry_headers.get("Retry-After") or retry_headers.get("retry-after")
        try:
            return max(1.0, min(float(retry_after), 300.0))
        except (TypeError, ValueError):
            return min(float(2 ** (attempt - 1)), 30.0)
