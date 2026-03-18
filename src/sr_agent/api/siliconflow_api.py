# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import yaml
import logging
import requests
from collections import defaultdict
from typing import Generator, Tuple, List, Dict
from .llm_api import LLMAPI
from ....utils import log_exception

_logger = logging.getLogger(__name__)


class SiliconFlowAPI(LLMAPI):
    supported_models = [
        "Qwen3-8B",
        "Deepseek-V3",
    ]

    def __init__(self, model='Qwen3-8B', max_tokens=1024, n=1, temperature=1.0, top_p=1.0):
        self.n = n
        self.top_p = top_p
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _request(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        url = 'https://api.siliconflow.cn/v1/chat/completions'
        headers = {
            'Authorization': f"Bearer {os.environ['SILICONFLOW_API_KEY']}",
            'Content-Type': 'application/json',
        }
        payload = {
            'messages': messages,
            'n': kwargs.get('n', self.n),
            'stop': [],
            'top_k': kwargs.get('top_k', 50),
            'top_p': kwargs.get('top_p', self.top_p),
            'min_p': kwargs.get('min_p', 0.05),
            'stream': kwargs.get('stream', False),
            'thinking_budget': kwargs.get('thinking_budget', 1024),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.5),
        }
        # Request the LLM API
        model = kwargs.get('model', self.model)
        if model == 'Qwen3-8B':
            results = yield from self.qwen3_8b(url, headers, payload, **kwargs)
        elif model == "Deepseek-V3":
            results = yield from self.deepseek_v3(url, headers, payload, **kwargs)
        else:
            raise ValueError(f"Model {model} not supported in SiliconFlowAPI.")
        return results

    def qwen3_8b(self, url, headers, payload, **kwargs) -> Generator[str, None, Dict]:
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
        for choice in responses["choices"]:
            yield choice["message"]["content"]
        usage = {'token': {}, 'price': {}}
        usage['token']['prompt'] = (prompt_tokens := responses["usage"]["prompt_tokens"])
        usage['token']['reason'] = (reason_tokens := responses["usage"]["completion_tokens_details"]["reasoning_tokens"])
        usage['token']['answer'] = (answer_tokens := responses["usage"]["completion_tokens"] - reason_tokens)
        if (other := responses["usage"]["total_tokens"] - prompt_tokens - answer_tokens - reason_tokens) != 0:
            usage['token']['other'] = other
        usage['price']['prompt'] = 0.00 * prompt_tokens / 1e6
        usage['price']['answer'] = 0.00 * answer_tokens / 1e6
        usage['price']['reason'] = 0.00 * reason_tokens / 1e6
        results = {
            'usage': usage,
            'messages': payload['messages'],
            'contents': [choice["message"]["content"] for choice in responses["choices"]],
            'responses': responses
        }
        return results

    def deepseek_v3(self, url, headers, payload, **kwargs) -> Generator[str, None, Dict]:
        ## Ensure this is a generator
        yield from []
        payload = {
            'model': 'deepseek-ai/DeepSeek-V3',
            **payload,
        }
        n, payload['n'] = payload['n'], 1  # Deepseek-V3 does not support n>1 in one request
        model = payload['model']
        usage = dict(token=defaultdict(float), price=defaultdict(float))
        responses = []
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
            responses.append(response)
            text = response["choices"][0]["message"]["content"]
            yield text
        results = {
            "usage": usage,
            "messages": payload["messages"], 
            'contents': [response['choices'][0]["message"]["content"] for response in responses],
            'responses': responses
        }
        return results
