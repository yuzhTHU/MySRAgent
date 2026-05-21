# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
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
        "deepseek/deepseek-v4-flash",
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
        if not self.tool_list:
            pass
        elif self.tool_parser:
            payload["messages"] = self.add_tool_description(payload["messages"])
        else:
            payload["tools"] = self.tool_description_json
            payload["tool_choice"] = "auto"

        # OpenRouter does not support `n` parameter now,
        # see https://github.com/OpenRouterTeam/openrouter-runner/issues/99
        details = []
        for idx in range(1, n + 1):
            try:
                completion = client.chat.completions.create(**payload)
                response_dict = completion.to_dict()
                message = completion.choices[0].message.to_dict()
                content = message['content'] or ""
            except Exception as e:
                _logger.error(f"Error requesting OpenRouterAPI({self.model}) since {log_exception(e)}")
                continue

            token_usage = {}
            price_usage = {}
            if usage := completion.usage:
                token_usage["prompt"] = usage.prompt_tokens
                token_usage["answer"] = usage.completion_tokens
                total_tokens = getattr(usage, "total_tokens", usage.prompt_tokens + usage.completion_tokens)
                if (other := total_tokens - usage.prompt_tokens - usage.completion_tokens) > 0:
                    token_usage["others"] = other
                price_usage['total'] = getattr(usage, "cost", 0)

            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            elif 'tool_calls' in message:
                tool_call = self.normalize_openai_tool_calls(message['tool_calls'])
            else:
                tool_call = []
                
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
