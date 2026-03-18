# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import logging
from copy import deepcopy
from openai import AzureOpenAI
from openai.types.responses import Response
from openai.types.chat import ChatCompletion
from collections import defaultdict
from typing import Generator, Tuple, List, Dict
from .llm_api import LLMAPI
from ....utils import log_exception

_logger = logging.getLogger(__name__)


class OpenAIAPI(LLMAPI):
    supported_models = [
        "gpt-4o-mini",
        "gpt-5-mini",
    ]

    def __init__(self, model='gpt-5-mini', max_tokens=4096, n=1, temperature=1.0, top_p=1.0, use_chat_completions=False):
        self.n = n
        self.top_p = top_p
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_chat_completions = use_chat_completions
        self.dummy_message = 'Please proofread the message above and say "I have received it."'

        if model == 'gpt-4o-mini' and not self.use_chat_completions:
            self.use_chat_completions = True
            _logger.warning("Turn to use create_chat_completions for gpt-4o-mini model to save token cost.")

    def _request(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        """
        Keyword Args:
            model (str): The model name to use. Default is 'gpt-4o-mini'.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
            temperature (float): Sampling temperature. Default is 1.0.
            top_p (float): Nucleus sampling probability. Default is 1.0.
            n (int): Number of completions to generate. Default is 1.
        """
        if self.use_chat_completions:
            results = yield from self.create_chat_completions(messages, **kwargs)
            return results
        else:
            results = yield from self.create_responses(messages, **kwargs)
            return results

    def create_responses(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        """ OpenAI 的最新 API, 建议新项目使用这个接口 (https://platform.openai.com/docs/guides/migrate-to-responses)
        但看起来它还缺了一些功能 (比如 n parameter), 而且也无法缓存 gpt-4o-mini 的 prompt token, 所以依然保留了 create_chat_completions 接口
        """
        ## Ensure this is a generator
        yield from []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        client = AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_ENDPOINT"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        payload = {
            "input": deepcopy(messages), # 因为后面可能要 pop, 所以 deepcopy 一下
            "reasoning": {"summary": "auto"},
            "model": (model := kwargs.get('model', self.model)),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            # "prompt_cache_retention": "24h"  # 官网上说有这个参数可以延长 cache 保存时间，但实际用起来会报错
            # "n": n,  # OpenAI Response API 不支持 parameter n
        }
        if payload['input'][0]['role'] == 'system':
            ## instruction 无法在多轮对话中保留，不好用
            # payload['instructions'] = payload['input'].pop(0)['content']
            ## 改用据说效果与 instruction 等价的 developer 角色
            payload['input'][0]['role'] = 'developer'

        n = kwargs.get('n', self.n)
        if n == 1:
            try:
                response = client.responses.create(**payload)
                yield response.output_text
                usage = self.parse_usage(response)
                results = {
                    'usage': usage,
                    'messages': messages,
                    'contents': response.output_text,
                    'response': response.to_dict(),
                }
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({model}) since: {log_exception(e)}")
                results = {"usage": {"token": {}, "price": {}}, "messages": messages, "contents": [], "response": None}
            finally:
                return results
        else:
            try:
                responses = []
                ## 构造 payload 将除最后一条 message 之外的所有前置信息输入模型
                assert payload['input'][-1]['role'] == 'user', "When n > 1, the last message must be from user."
                last_message = payload['input'].pop(-1)
                payload['input'].append({'role': 'user', 'content': self.dummy_message})
                response = client.responses.create(**payload)
                usage = self.parse_usage(response)
                responses.append(response)
                ## 构造 child payload 多次输入最后一条 message 以获得多条回复并尽量增加 cached input token 的命中率
                child_payload = deepcopy(payload)
                child_payload['previous_response_id'] = response.id
                child_payload['input'] = [last_message]
                for _ in range(n):
                    child_response = client.responses.create(**child_payload)
                    responses.append(child_response)
                    yield child_response.output_text
                    new_usage = self.parse_usage(child_response)
                    for k, v in new_usage['token'].items():
                        usage['token'][k] = usage['token'].get(k, 0) + v
                    for k, v in new_usage['price'].items():
                        usage['price'][k] = usage['price'].get(k, 0) + v
                results = {
                    'usage': usage,
                    'messages': messages,
                    'contents': [resp.output_text for resp in responses[1:]],  # 排除第一条前置信息的回复
                    'response': [resp.to_dict() for resp in responses],
                }
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({model}) since: {log_exception(e)}")
                results = {"usage": {"token": {}, "price": {}}, "messages": messages, "contents": [], "response": None}
            finally:
                return results

    def create_chat_completions(self, messages: List|str, **kwargs) -> Generator[str, None, Dict]:
        """ OpenAI 的旧版 API, 建议新项目使用 create_responses 接口 """
        ## Ensure this is a generator
        yield from []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        client = AzureOpenAI(
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ["OPENAI_OLDTIME_ENDPOINT"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        payload = {
            "messages": messages,
            "n": kwargs.get('n', self.n),
            "model": (model := kwargs.get('model', self.model)),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
        }
        assert model == 'gpt-4o-mini', "Only gpt-4o-mini model is supported in chat_completions."

        try:
            response = client.chat.completions.create(**payload)
            for choice in response.choices:
                yield choice.message.content
            usage = self.parse_chat_completions_usage(response)
            results = {
                'usage': usage,
                'messages': messages,
                'contents': [choice.message.content for choice in response.choices],
                'response': response.to_dict(),
            }
        except Exception as e:
            _logger.error(f"Error requesting {type(self).__name__}({model}) since: {log_exception(e)}")
            results = {"usage": {"token": {}, "price": {}}, "messages": messages, "contents": [], "response": None}
        finally:
            return results

    def parse_usage(self, response: Response) -> Dict:
        usage = {'token': defaultdict(float), 'price': defaultdict(float)}
        if response.model.startswith('gpt-5-mini'):
            usage['token']['cached'] = (cached_tokens := response.usage.input_tokens_details.cached_tokens)
            usage['token']['prompt'] = (prompt_tokens := response.usage.input_tokens - cached_tokens)
            usage['token']['reason'] = (reason_tokens := response.usage.output_tokens_details.reasoning_tokens)
            usage['token']['answer'] = (answer_tokens := response.usage.output_tokens - reason_tokens)
            if (other := response.usage.total_tokens - cached_tokens - prompt_tokens - reason_tokens - answer_tokens) != 0:
                usage['token']['other'] = other
            usage['price']['cached'] = 0.025 * cached_tokens / 1e6
            usage['price']['prompt'] = 0.25  * prompt_tokens / 1e6
            usage['price']['reason'] = 2.0   * reason_tokens / 1e6
            usage['price']['answer'] = 2.0   * answer_tokens / 1e6
        elif response.model.startswith('gpt-4o-mini'):
            usage['token']['prompt'] += (prompt_tokens := response.usage.prompt_tokens)
            usage['token']['reason'] += (reason_tokens := response.usage.completion_tokens_details.reasoning_tokens)
            usage['token']['answer'] += (answer_tokens := response.usage.completion_tokens - reason_tokens)
            if (other := response.usage.total_tokens - cached_tokens - prompt_tokens - reason_tokens - answer_tokens) != 0:
                usage['token']['other'] = other
            usage['price']['prompt'] += 0.15 * prompt_tokens / 1e6
            usage['price']['reason'] += 0.60 * reason_tokens / 1e6
            usage['price']['answer'] += 0.60 * answer_tokens / 1e6
        else:
            raise NotImplementedError(f"Usage parsing for model {response.model} is not implemented.")
        return usage

    def parse_chat_completions_usage(self, response: ChatCompletion) -> Dict:
        usage = {'token': defaultdict(float), 'price': defaultdict(float)}
        if response.model.startswith('gpt-4o-mini'):
            usage['token']['prompt'] += (prompt_tokens := response.usage.prompt_tokens)
            usage['token']['reason'] += (reason_tokens := response.usage.completion_tokens_details.reasoning_tokens)
            usage['token']['answer'] += (answer_tokens := response.usage.completion_tokens - reason_tokens)
            if (other := response.usage.total_tokens - prompt_tokens - reason_tokens - answer_tokens) > 0:
                usage['token']['other'] = other
            usage['price']['prompt'] += 0.15 * prompt_tokens / 1e6
            usage['price']['reason'] += 0.60 * reason_tokens / 1e6
            usage['price']['answer'] += 0.60 * answer_tokens / 1e6
        else:
            raise NotImplementedError(f"Usage parsing for model {response.model} is not implemented.")
        return usage
