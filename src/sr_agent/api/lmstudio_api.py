# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Generator, List
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

from .llm_api import LLMAPI
from ..utils import log_exception

_logger = logging.getLogger(f"sr_agent.{__name__}")


class LMStudioAPI(LLMAPI):
    """LLM API adapter for an LM Studio server.

    ``LMSTUDIO_ENDPOINT`` may point at LM Studio's native ``/api/v1/chat``
    endpoint, as recommended by LM Studio. SRAgent needs multi-turn messages
    and custom function tools, so requests are sent to the OpenAI-compatible
    ``/v1/chat/completions`` endpoint on the same server.
    """

    supported_models = ["qwen_qwen3-4b-instruct-2507"]

    def __init__(self, model: str = "qwen_qwen3-4b-instruct-2507", **kwargs):
        super().__init__(model=model, **kwargs)

    @staticmethod
    def normalize_endpoint(endpoint: str) -> str:
        """Return the OpenAI-compatible chat-completions URL."""
        endpoint = endpoint.strip()
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError(f"Invalid LMSTUDIO_ENDPOINT: {endpoint!r}")

        path = parts.path.rstrip("/").removesuffix("/api/v1/chat").removesuffix("/v1/chat/completions") + "/v1/chat/completions"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _request(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Generator[str, None, Dict]:
        yield from []
        load_dotenv()

        endpoint = self.normalize_endpoint(os.environ["LMSTUDIO_ENDPOINT"])
        api_key = os.environ["LMSTUDIO_API_KEY"]
        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if not self.tool_list:
            pass
        elif self.tool_parser:
            payload["messages"] = self.add_tool_description(payload["messages"])
        else:
            payload["tools"] = self.tool_description_json
            payload["tool_choice"] = "auto"

        # LM Studio is normally on the local network. Ignore shell proxy
        # variables so a loopback proxy cannot intercept or block the request.
        session = requests.Session()
        session.trust_env = False
        timeout = float(os.environ.get("LMSTUDIO_TIMEOUT", "300"))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        max_retry = 3
        for attempt in range(1, max_retry + 1):
            try:
                response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                response_dict = response.json()
                choices = response_dict.get("choices") or []
                if not choices:
                    raise ValueError(f"LMStudioAPI({self.model}) returned no choices.")
                break
            except Exception as error:
                if attempt == max_retry:
                    raise RuntimeError(
                        f"LMStudioAPI({self.model}) failed after {max_retry} attempts."
                    ) from error
                delay = min(float(2 ** (attempt - 1)), 30.0)
                _logger.error(
                    f"Error requesting LMStudioAPI({self.model}) since "
                    f"{log_exception(error, with_traceback=False)}"
                )
                _logger.info(
                    f"Retrying LMStudioAPI({self.model}) in {delay:g}s "
                    f"(attempt {attempt}/{max_retry})."
                )
                time.sleep(delay)

        details = []
        for choice in choices:
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            else:
                tool_call = self.normalize_openai_tool_calls(message.get("tool_calls") or [])
                message["tool_calls"] = [call.raw for call in tool_call]
            if not (content.strip() or tool_call):
                _logger.warning(f"LMStudioAPI({self.model}) returned an empty choice.")
            details.append({"content": content, "tool_call": tool_call, "message": message})
            yield {"content": content, "tool_call": tool_call, "message": message}

        raw_usage = response_dict.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        reasoning_tokens = int(
            (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        token_usage = {
            "prompt": prompt_tokens,
            "answer": max(completion_tokens - reasoning_tokens, 0),
        }
        if reasoning_tokens:
            token_usage["reason"] = reasoning_tokens
        total_tokens = int(raw_usage.get("total_tokens") or prompt_tokens + completion_tokens)
        accounted = prompt_tokens + completion_tokens
        if total_tokens > accounted:
            token_usage["other"] = total_tokens - accounted

        return {
            "usage": {"token": token_usage, "price": {"total": 0.0}},
            "contents": [detail["content"] for detail in details],
            "response_message": details[0]["content"] if details else "",
            "tool_calls": (
                details[0]["tool_call"]
                if len(details) == 1
                else [detail["tool_call"] for detail in details]
            ),
            "responses": [response_dict],
        }
