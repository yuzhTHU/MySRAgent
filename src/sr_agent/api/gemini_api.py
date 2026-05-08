# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from collections import defaultdict
from typing import Generator, List, Dict
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types
from .llm_api import LLMAPI

_logger = logging.getLogger(__name__)


class GeminiAPI(LLMAPI):
    supported_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    def __init__(self, model='gemini-2.5-pro', **kwargs):
        super().__init__(model=model, **kwargs)

    def _request(self, messages: List[Dict[str, str]], n=1):
        ## Ensure this is a generator
        yield from []
        if not self.tool_list:
            pass
        elif self.tool_parser:
            messages = self.add_tool_description(messages)
        else:
            raise NotImplementedError("GeminiAPI does not support parser='openai' native tool calls.")
        api_key = os.environ.get("GEMINI_API_KEY", None)
        self.setup_proxy()
        config = types.GenerateContentConfig(
            candidate_count=n,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=True,
            ),
        )
        if self.model in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
            config.thinking_config = None
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=messages,
            config=config,
        )
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        reason_tokens = usage.thoughts_token_count or 0
        answer_tokens = usage.total_token_count - prompt_tokens - reason_tokens
        token_usage = {
            "prompt": prompt_tokens,
            "answer": answer_tokens,
            "reason": reason_tokens,
        }
        price_usage = {"total": 0.0}
        details = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if not (content := part.text):
                    continue
                elif part.thought:
                    continue
                else:
                    if not self.tool_list:
                        tool_call = []
                    elif not self.tool_parser:
                        _logger.warning("Received tool call but no tool parser is set. Ignoring tool call.")
                        tool_call = []
                    else:
                        tool_call = self.tool_parser.parse_response(content)
                    message = {"role": "assistant", "content": content}
                    details.append({
                        "content": content,
                        "tool_call": tool_call,
                        "token_usage": usage,
                        "price_usage": price_usage,
                        "response": candidate.to_json_dict(),
                        "message": message,
                    })
                    yield {"content": content, "tool_call": tool_call, "message": message}
                    break
        token_usage = defaultdict(float)
        for detail in details:
            for key, value in detail["token_usage"].items():
                token_usage[key] += value
        price_usage = defaultdict(float)
        for detail in details:
            for key, value in detail["price_usage"].items():
                price_usage[key] += value
        return {
            "usage": {"token": token_usage, "price": price_usage},
            "contents": [detail["content"] for detail in details],
            "tool_calls": [detail["tool_call"] for detail in details],
            "responses": [detail["response"] for detail in details],
        }
