# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from openai import OpenAI
from collections import defaultdict
from typing import Generator, List, Dict
from .llm_api import LLMAPI
from ..utils import log_exception

_logger = logging.getLogger(__name__)


class DeepSeekAPI(LLMAPI):
    supported_models = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    def __init__(self, model='deepseek-chat', **kwargs):
        super().__init__(model=model, **kwargs)

    def _request(self, messages: List[Dict[str, str]], n=1, max_tokens=1024) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        payload = {
            "model": self.model,
            "messages": messages, 
            "stream": False, 
            "max_tokens": max_tokens,
        }
        if not self.tool_list:
            pass
        elif self.tool_parser:
            payload['messages'] = self.add_tool_description(payload['messages'])
        else:
            payload["tools"] = self.tool_description_json
            payload["tool_choice"] = "auto"

        details = []
        for idx in range(1, n + 1):
            try:
                response = client.chat.completions.create(**payload)
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({self.model}) since {log_exception(e)}")
                continue
            response_dict = response.to_dict()

            token_usage = {}
            token_usage['prompt'] = (prompt_tokens := response.usage.prompt_tokens)
            token_usage['answer'] = (answer_tokens := response.usage.completion_tokens)
            if (other := response.usage.total_tokens - prompt_tokens - answer_tokens) > 0:
                token_usage['other'] = other
            price_usage = {}
            price_usage['prompt'] = 0.28 / 1e6 * token_usage['prompt']
            price_usage['answer'] = 0.42 / 1e6 * token_usage['answer']
            price_usage['other']  = 0.42 / 1e6 * token_usage['other']

            message = response.choices[0].message
            content = message.content or ""
            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            else:
                tool_call = self.normalize_openai_tool_calls(message.tool_calls)

            token_usage = {"prompt": prompt_tokens, "answer": answer_tokens}
            details.append({
                "content": content,
                "tool_call": tool_call,
                "token_usage": token_usage,
                "price_usage": price_usage,
                "response": response_dict,
            })
            yield {'content': content, 'tool_call': tool_call, 'message': message}

        token_usage = defaultdict(float)
        for detail in details:
            for key, value in detail['token_usage'].items():
                token_usage[key] += value
        price_usage = defaultdict(float)
        for detail in details:
            for key, value in detail['price_usage'].items():
                price_usage[key] += value
        return {
            "usage": {"token": token_usage, "price": price_usage},
            "contents": [detail['content'] for detail in details],
            "tool_calls": [detail['tool_call'] for detail in details],
            "responses": [detail['response'] for detail in details],
        }
