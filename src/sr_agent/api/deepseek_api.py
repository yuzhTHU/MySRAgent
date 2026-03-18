# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from openai import OpenAI
from collections import defaultdict
from typing import Generator, List, Dict
from .llm_api import LLMAPI
from ....utils import log_exception

_logger = logging.getLogger(__name__)


class DeepSeekAPI(LLMAPI):
    supported_models = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    def __init__(self, model='deepseek-chat', max_tokens=1024, n=1):
        self.n = n
        self.model = model
        self.max_tokens = max_tokens
        # deepseek-reasoner needs more tokens for reasoning
        if model == 'deepseek-reasoner':
            self.max_tokens = 4096

    def _request(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        """
        Keyword Args:
            model (str): The model name to use. Default is 'deepseek-chat'.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
            n (int): Number of completions to generate. Default is 1.
        """
        ## Ensure this is a generator
        yield from []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        payload = {
            "model": (model := kwargs.get('model', self.model)), 
            "messages": messages, 
            "stream": False, 
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
        }
        usage = dict(token=defaultdict(float), price=defaultdict(float))
        responses = []
        for _ in range(kwargs.get('n', self.n)):
            try:
                response = client.chat.completions.create(**payload)
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({model}) since: {log_exception(e)}")
                continue
            usage['token']['prompt'] += (prompt_tokens := response.usage.prompt_tokens)
            usage['token']['answer'] += (answer_tokens := response.usage.completion_tokens)
            if (other := response.usage.total_tokens - prompt_tokens - answer_tokens) > 0:
                usage['token']['other'] += other
            usage['price']['prompt'] += (prompt_price := 0.28 * response.usage.prompt_tokens / 1e6)
            usage['price']['answer'] += (answer_price := 0.42 * response.usage.completion_tokens / 1e6)
            responses.append(response.to_dict())
            text = response.choices[0].message.content
            yield text
        results = {
            "usage": usage,
            "messages": messages, 
            "contents": [response['choices'][0]['message']['content'] for response in responses],
            "responses": responses, 
        }
        return results
