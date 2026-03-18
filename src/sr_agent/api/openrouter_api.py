# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from openai import OpenAI
from typing import Generator, Tuple, List, Dict
from .llm_api import LLMAPI

_logger = logging.getLogger(__name__)


class OpenRouterAPI(LLMAPI):
    supported_models = [
        "kimi-k2",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    def _request(self, prompt: str) -> Generator[Tuple[str, dict], None, List | Dict]:
        raise DeprecationWarning
        yield from []
        api_key = os.environ["OPENROUTER_API_KEY"]
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        if self.model in ["gemini-2.5-pro", "gemini-2.5-flash"]:
            prompt = prompt.replace(
                "Only return the `def get_equations(...)` function, put your thinkings and explanations in the docstring of the function or as comments in the code (between `def` and `return`).",
                "Only return the `def get_equations(...)` function and very brief explanations in the docstring, without complex thinkings or explanations.",
            )
        payload = {"messages": [{"role": "user", "content": prompt}]}
        if self.model == "kimi-k2":
            payload["model"] = "moonshotai/kimi-k2"
        elif self.model == "gemini-2.5-pro":
            payload["model"] = "google/gemini-2.5-pro"
        elif self.model == "gemini-2.5-flash":
            payload["model"] = "google/gemini-2.5-flash"
        else:
            payload["model"] = self.model
        if self.model == "gemini-2.5-pro":
            payload["extra_body"] = {"reasoning": {"max_tokens": 128}}
        # OpenRouter does not support `n` parameter now,
        # see https://github.com/OpenRouterTeam/openrouter-runner/issues/99
        results = []
        for _ in range(self.generate_per_prompt):
            try:
                completion = client.chat.completions.create(**payload)
                results.append(completion.to_dict())
                answer = completion.choices[0].message.content
                usage = {
                    "total": completion.usage.total_tokens,
                    "prompt": completion.usage.prompt_tokens,
                    "answer": completion.usage.completion_tokens,
                }
                yield answer, usage
            except Exception as e:
                _logger.error(f"Error requesting {self.model}: {e}")
        return {"prompt": prompt, "response": results}
