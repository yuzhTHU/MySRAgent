# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""基类 Agent 和 Buffer 的定义。

提供符号回归 Agent 的基础框架，包括主循环 Pipeline 和工具调用机制。
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict, List, Tuple
from .api.llm_api import LLMAPI
from .tools import BaseTool
from .parser import BaseParser
from .utils import FactoryMixin, ParallelTimer, NamedTimer

_logger = logging.getLogger(f'sr_agent.{__name__}')


class SRAgent(FactoryMixin):
    """符号回归 Agent。

    提供完整的符号回归流程框架，包括数据预处理、Prompt 生成、LLM 请求、
    工具调用、Buffer 更新等。具体实现需继承此类并实现必要方法。

    Attributes:
        llm_api: LLM API 实例。
        buffer: 对话历史 Buffer。
        tools: 可用工具字典。
        max_iteration: 最大迭代次数。
    """

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        tools: List[str] | None = None,
        max_iteration: int = 100,
        verbose: bool = False,
        tool_parser: str | BaseParser = 'text',
    ):
        """初始化 Agent。

        Args:
            llm_provider: LLM 提供商名称（如 "openai", "siliconflow"）。
            llm_model: 模型名称（如 "gpt-4o-mini"）。
            tools: 可用工具名称列表。None 表示使用全部工具。
            max_iteration: 最大迭代次数，默认 100。
            verbose: 是否启用详细日志。
            tool_parser: 工具解析器，可以是字符串（'text', 'json'）或 BaseParser 实例。
        """
        self.llm_api = LLMAPI.load(llm_provider, llm_model)
        self.buffer = []
        self.tools = [tool for tool in BaseTool.load_tool_list() if tools is None or tool['name'] in tools]
        self.max_iteration = max_iteration
        self.named_timer = NamedTimer()
        self.token_counter = ParallelTimer(unit='token')
        self.money_counter = ParallelTimer(unit='$')
        self.tool_call_counter = ParallelTimer(unit='call')
        self.tool_context = {}

        # 初始化工具解析器
        if isinstance(tool_parser, BaseParser):
            self.tool_parser = tool_parser
        else:
            self.tool_parser = BaseParser.create(tool_parser, tool_list=tools)

        if verbose:
            _logger.setLevel(logging.DEBUG)
            for handler in _logger.handlers:
                handler.setLevel(logging.DEBUG)
            _logger.debug("Verbose mode enabled, set logging level to DEBUG")
        _logger.info(f"Initialized {self.__class__.__name__} with model {llm_model}")

    def fit(
        self,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        problem_description: str,
    ) -> Dict[str, Any]:
        """执行符号回归任务的主入口。

        包含以下阶段：
        1. 初始化：设置数据、重置状态
        2. 主循环：生成 Prompt → 请求 LLM → 解析 Action → 执行工具 → 更新 Buffer
        3. 后处理：整理结果
        4. 日志打印和终止检查

        Args:
            X: 输入特征字典，键为特征名，值为 numpy 数组。
            y: 目标变量 numpy 数组。
            problem_description: 问题描述字符串，告知 Agent 任务目标。

        Returns:
            包含最终结果的字典，通常包括：
            - best_formula: 最优公式
            - best_score: 最优得分
            - history: 历史迭代记录
        """
        _logger.info(f"Starting fit with {len(X)} features and {len(y)} samples")
        _logger.info(f"Problem description: {problem_description}")

        # 初始化 Buffer
        self.buffer.clear()

        # 构建 system prompt - 告知 LLM 它的角色和目标
        system_prompt = """You are a Symbolic Regression Agent. Your goal is to discover mathematical formulas that explain the relationship between input features (X) and target variable (y).

"""
        # 使用工具解析器生成工具描述和输出格式
        tool_description = self.tool_parser.format_tools()
        system_prompt += tool_description

        self.buffer.append({
            "role": "system",
            "content": system_prompt
        })

        # 构建 user prompt - 告知具体问题和数据信息
        user_prompt = f"""{problem_description}

## Data Information:
- Number of samples: {len(y)}
- Number of features: {len(X)}
- Feature names: {list(X.keys())}

Please start by analyzing the data to understand the relationship between features and target."""

        self.buffer.append({
            "role": "user",
            "content": user_prompt
        })

        # 初始化 Tool Context
        self.tool_context['x'] = X
        self.tool_context['y'] = y

        # 开始迭代
        history = []
        best_record = None
        best_score = float('inf')
        best_formula = None
        for iteration in range(self.max_iteration):
            _logger.info(f"Start Iteration {iteration + 1}/{self.max_iteration}")

            # Step 1: 根据 Buffer 创建 Prompt
            messages = self.build_prompt()
            _logger.debug(f"Built prompt with {len(messages)} messages")

            # Step 2: 请求 LLM，得到 Response
            llm_response = self.request_llm(messages)
            _logger.debug(f"LLM response: {llm_response[:200]}...")

            # Step 3: 解析 Response，得到 Action
            action = self.parse_actions(llm_response)
            # 确保至少有一个 action
            if not action:
                action = [('response', {'content': llm_response})]
            _logger.info(f"Parsed action: {action[0]}")

            # Step 4: 执行 Action，得到 Result
            result = self.execute_action(action)
            _logger.debug(f"Action result: {result}")

            # Step 5: 基于 Action 和 Result 更新 Buffer
            self.update_buffer(action, result)

            # 记录历史
            history.append({
                "iteration": iteration + 1,
                "action": action,
                "result": result,
            })

            # 检查终止条件
            if self._should_stop(iteration, action, result):
                _logger.info("Termination condition met")
                break

            # 更新最优结果
            for act, res in zip(action, result):
                if res is not None and isinstance(res, dict):
                    if res.get('success') and res.get('mse') is not None:
                        if res['mse'] < best_score:
                            best_score = res['mse']
                            best_formula = res.get('formula')
                            best_record = {
                                "formula": best_formula,
                                "score": best_score,
                                "mse": res.get('mse'),
                                "rmse": res.get('rmse'),
                                "mae": res.get('mae'),
                                "r2": res.get('r2'),
                            }

            # 统计本轮工具调用次数
            tool_calls_this_round = {}
            for name, _ in action:
                if name not in ('think', 'response'):
                    tool_calls_this_round[name] = tool_calls_this_round.get(name, 0) + 1

            # 打印本轮日志
            tool_calls_str = "; ".join(f"{n}(+{c})" for n, c in sorted(tool_calls_this_round.items())) if tool_calls_this_round else "0"
            log = {
                "Best": f"{best_formula} (MSE={best_score:.6g})" if best_formula else "None",
                "Tool Calls": tool_calls_str,
                "Speed": self.token_counter.to_str('count', 'speed', 'by_count'),
                "Time Usage": self.named_timer.to_str('pace', 'time', 'by_time'),
                "Token Usage": self.token_counter.to_str('count', None, None),
                "Price Usage": self.money_counter.to_str('count', None, None),
            }
            msg = " | ".join(f"\033[4m{k}\033[0m={v}" for k, v in log.items())
            _logger.info(msg)

        # ========== 后处理阶段 ==========
        _logger.info(f"Fit completed. Best formula: {best_formula}, Best score: {best_score}")

        return {
            "best_formula": best_formula,
            "best_score": best_score,
            "history": history,
            "iterations": len(history),
        }

    def build_prompt(self) -> List[Dict[str, str]]:
        """根据当前 Buffer 创建 LLM Prompt。

        Returns:
            格式化后的消息列表。
        """
        return self.buffer

    def request_llm(self, messages: List[Dict[str, str]]) -> str:
        """向 LLM 发送请求并获取响应。

        Args:
            messages: 格式化的消息列表。

        Returns:
            LLM 返回的文本内容。
        """
        for content in (llm_result := self.llm_api(messages)):
            break
        for name, num in llm_result.usage['token'].items():
            self.token_counter.add(name, num)
        for name, num in llm_result.usage['price'].items():
            self.money_counter.add(name, num)
        # if self.save_response:
        #     with open(Path(self.save_path) / 'response.json')
        return content

    def parse_actions(self, response: str) -> List[Tuple[str, Any]]:
        """解析 LLM 返回，提取 Actions。

        使用 self.tool_parser 进行解析。

        Args:
            response: LLM 的原始响应文本。

        Returns:
            工具调用列表，每个元素为 (tool_name, params) 元组。
        """
        return self.tool_parser.parse_response(response)

    def execute_action(self, actions: List[Tuple[str, Any]]) -> List[Dict|None]:
        """执行 Action。

        Args:
            actions: 多个 (name, value) 元组构成的数组

        Returns:
            执行结果列表。对于不需要执行的 name（例如 think 或者 response）返回 None
        """
        results = []
        for idx, (name, params) in enumerate(actions):
            if name == 'think':
                results.append(None)
            elif name == 'response':
                results.append(None)
            elif name in [tool['name'] for tool in self.tools]:
                self.tool_call_counter.add(name)
                tool = BaseTool.create(name, **self.tool_context)
                result = tool(**params)
                results.append(result)
            else:
                results.append({'success': False, 'error': f'Unknown tool calling for "{name}"'})

        return results

    def update_buffer(self, actions: List[Tuple[str, Any]], results: List[Dict|None]) -> None:
        """根据 Action 和结果更新 Buffer。

        Args:
            actions: 执行的 Action。
            results: 执行结果。
        """
        for (name, params), result in zip(actions, results):
            if name == 'think':
                self.buffer.append({'role': 'assistant', 'content': f"Thought: {params.get('content', '')}"})
            elif name == 'response':
                self.buffer.append({'role': 'assistant', 'content': f"Final Answer: {params.get('content', '')}"})
            else:
                # 工具调用
                self.buffer.append({'role': 'assistant', 'content': f"Action: {name}({params})"})
                self.buffer.append({'role': 'user', 'content': str(result)})

    def _should_stop(self, iteration: int, actions: List[Tuple[str, Any]], results: List[Dict|None]) -> bool:
        """检查是否应该终止迭代。

        Args:
            iteration: 当前迭代次数。
            actions: 执行的 Action 列表。
            results: 执行结果列表。

        Returns:
            是否应该终止。
        """
        # 检查是否有明确的 Final Answer（不是默认的 response）
        for action, result in zip(actions, results):
            if action[0] == 'response':
                content = action[1].get('content', '') if isinstance(action[1], dict) else ''
                # 只有包含 "Final Answer" 标记的才算真正的终止信号
                if 'Final Answer' in content or 'final answer' in content.lower():
                    return True

        # 检查是否达到了很好的分数（例如 MSE < 1e-6）
        for result in results:
            if result is not None and isinstance(result, dict):
                if result.get('success') and result.get('mse') is not None:
                    if result['mse'] < 1e-10:
                        return True

        return False
