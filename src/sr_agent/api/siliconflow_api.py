# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
import requests
from dotenv import load_dotenv
from collections import defaultdict
from typing import Generator, List, Dict
from .llm_api import LLMAPI
from ..utils import log_exception

_logger = logging.getLogger(__name__)


class SiliconFlowAPI(LLMAPI):
    supported_models = [
        "Qwen3-8B",
        "Deepseek-V3",
    ]

    def __init__(self, model='Qwen3-8B', **kwargs):
        super().__init__(model=model, **kwargs)

    def _request(
        self,
        messages: List[Dict[str, str]],
        n=1,
        max_tokens=1024,
        temperature=1.0,
        top_p=1.0,
    ) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        load_dotenv()
        url = 'https://api.siliconflow.cn/v1/chat/completions'
        headers = {
            'Authorization': f"Bearer {os.environ['SILICONFLOW_API_KEY']}",
            'Content-Type': 'application/json',
        }
        payload = {
            'messages': messages,
            'n': n,
            'stop': [],
            'top_k': 50,
            'top_p': top_p,
            'min_p': 0.05,
            'stream': False,
            'thinking_budget': 1024,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'frequency_penalty': 0.5,
        }
        if not self.tool_list:
            pass
        elif self.tool_parser:
            payload["messages"] = self.add_tool_description(payload["messages"])
        else:
            payload["tools"] = self.tool_description_json
            payload["tool_choice"] = "auto"
        # Request the LLM API
        model = self.model
        if model == 'Qwen3-8B':
            results = yield from self.qwen3_8b(url, headers, payload)
        elif model == "Deepseek-V3":
            results = yield from self.deepseek_v3(url, headers, payload)
        else:
            raise ValueError(f"Model {model} not supported in SiliconFlowAPI.")
        return results

    def qwen3_8b(self, url, headers, payload) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        payload = {
            'model': 'Qwen/Qwen3-8B',
            'enable_thinking': True,
            **payload
        }
        try:
            res = requests.request("POST", url, json=payload, headers=headers)
        except Exception as e:
            _logger.error(f"Error requesting {self.model}: {log_exception(e)}")
            return []
        if res.status_code != 200:
            _logger.error(f"Error requesting {self.model}: {res.text}")
            return []
        responses = res.json()
        details = []
        for choice in responses["choices"]:
            content = choice["message"]["content"]
            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            else:
                tool_call = self.normalize_openai_tool_calls(choice["message"].get("tool_calls"))
            details.append({
                "content": content,
                "tool_call": tool_call,
                "token_usage": {},
                "price_usage": {},
                "response": {"choices": [choice]},
            })
            yield {"content": content, "tool_call": tool_call, "message": choice["message"]}
        usage = {'token': {}, 'price': {}}
        usage['token']['prompt'] = (prompt_tokens := responses["usage"]["prompt_tokens"])
        usage['token']['reason'] = (reason_tokens := responses["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0))
        usage['token']['answer'] = (answer_tokens := responses["usage"]["completion_tokens"] - reason_tokens)
        if (other := responses["usage"]["total_tokens"] - prompt_tokens - answer_tokens - reason_tokens) != 0:
            usage['token']['other'] = other
        usage['price']['prompt'] = 0.00 * prompt_tokens / 1e6
        usage['price']['answer'] = 0.00 * answer_tokens / 1e6
        usage['price']['reason'] = 0.00 * reason_tokens / 1e6
        for detail in details:
            detail["token_usage"] = {"prompt": prompt_tokens, "answer": answer_tokens, "reason": reason_tokens}
            detail["price_usage"] = dict(usage["price"])
        return {
            "usage": usage,
            "contents": [detail["content"] for detail in details],
            "response_message": details[0]["content"] if details else "",
            "tool_calls": details[0]["tool_call"] if len(details) == 1 else [detail["tool_call"] for detail in details],
            "responses": [responses],
        }

    def deepseek_v3(self, url, headers, payload) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        payload = {
            'model': 'deepseek-ai/DeepSeek-V3',
            **payload,
        }
        n, payload['n'] = payload['n'], 1  # Deepseek-V3 does not support n>1 in one request
        model = payload['model']
        usage = dict(token=defaultdict(float), price=defaultdict(float))
        details = []
        for _ in range(n):
            try:
                res = requests.request("POST", url, json=payload, headers=headers)
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({model}): {log_exception(e)}")
                continue
            if res.status_code != 200:
                _logger.error(f"Error requesting {type(self).__name__}({model}): {res.text}")
                continue
            response = res.json()
            usage['token']['prompt'] += (prompt_tokens := response["usage"]["prompt_tokens"])
            usage['token']['answer'] += (answer_tokens := response["usage"]["completion_tokens"])
            if (other := response["usage"]["total_tokens"] - prompt_tokens - answer_tokens) > 0:
                usage['token']['other'] += other
            usage['price']['prompt'] += 2/7.0 * prompt_tokens / 1e6 # 7.0 CNY ~ 1.0 USD
            usage['price']['answer'] += 8/7.0 * answer_tokens / 1e6
            message = response["choices"][0]["message"]
            content = message["content"] or ""
            if not self.tool_list:
                tool_call = []
            elif self.tool_parser:
                tool_call = self.tool_parser.parse_response(content)
            else:
                tool_call = self.normalize_openai_tool_calls(message.get("tool_calls"))
            details.append({
                "content": content,
                "tool_call": tool_call,
                "token_usage": {"prompt": prompt_tokens, "answer": answer_tokens},
                "price_usage": {
                    "prompt": 2/7.0 * prompt_tokens / 1e6,
                    "answer": 8/7.0 * answer_tokens / 1e6,
                },
                "response": response,
            })
            yield {"content": content, "tool_call": tool_call, "message": message}
        return {
            "usage": usage,
            "contents": [detail["content"] for detail in details],
            "response_message": details[0]["content"] if details else "",
            "tool_calls": details[0]["tool_call"] if len(details) == 1 else [detail["tool_call"] for detail in details],
            "responses": [detail["response"] for detail in details],
        }
