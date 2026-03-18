# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from typing import Generator, Tuple, List, Dict
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

    def __init__(self, model='gemini-2.5-pro'):
        self.model = model

    def _request(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        model = kwargs
        api_key = os.environ.get("GEMINI_API_KEY", None)
        os.environ["HTTP_PROXY"] = os.environ.get("MY_HTTP_PROXY", None)
        os.environ["HTTPS_PROXY"] = os.environ.get("MY_HTTPS_PROXY", None)
        config = types.GenerateContentConfig(
            candidate_count=self.generate_per_prompt,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=True,
            ),
        )
        if self.model == "gemini-2.5-pro":
            prompt = prompt.replace(
                "Only return the `def get_equations(...)` function, put your thinkings and explanations in the docstring of the function or as comments in the code (between `def` and `return`).",
                "Only return the `def get_equations(...)` function and very brief explanations in the docstring, without complex thinkings or explanations.",
            )
            prompt = f"{prompt}\n\n(You are limited to thinking for 128 tokens.)"
            config.thinking_config.thinking_budget = 128
        if self.model in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
            config.thinking_config = None
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            model_list = self.supported_models
            idx = model_list.index(self.model)
            self.model = model_list[(idx + 1) % len(model_list)]
            _logger.error(
                f"Error requesting {self.model}, switching to {self.model}: {e}"
            )
            yield from []
            return
        finally:
            if self._model != self.model:
                # Check if the model has been used today
                now = datetime.now(timezone(timedelta(hours=-7)))
                date = now.strftime("%Y-%m-%d")
                if date not in self._date_list:
                    self._date_list.append(date)
                    self.model = self._model

        usage = response.usage_metadata
        total_tokens = usage.total_token_count
        prompt_tokens = usage.prompt_token_count
        reason_tokens = usage.thoughts_token_count or 0
        answer_tokens = total_tokens - prompt_tokens - reason_tokens
        usage = {
            "total": total_tokens,
            "prompt": prompt_tokens,
            "answer": answer_tokens,
            "reason": reason_tokens,
        }
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if not part.text:
                    continue
                if part.thought:
                    continue
                else:
                    yield part.text, usage
                    break
            usage = {k: 0 for k in usage}  # Reset usage for next candidate
        return {"prompt": prompt, "response": response.to_json_dict()}
