# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""基类 Agent 和 Buffer 的定义。

提供符号回归 Agent 的基础框架，包括主循环 Pipeline 和工具调用机制。
"""
from __future__ import annotations
import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from .api.llm_api import LLMAPI
from .parser import BaseParser
from .api.core import ToolCall
from .tools import BaseTool, ToolCallResult
from .utils import FactoryMixin, ParallelTimer, NamedTimer
from .utils.logger import setup_logging
from .utils import render_python, render_markdown, tag2ansi

_logger = logging.getLogger(f'sr_agent.{__name__}')


class SRAgent(FactoryMixin):
    """符号回归 Agent。

    提供完整的符号回归流程框架，包括数据预处理、Prompt 生成、LLM 请求、
    工具调用、Buffer 更新等。具体实现需继承此类并实现必要方法。

    Attributes:
        llm_api: LLM API 实例。
        buffer: 对话历史 Buffer。
        max_refinement_depth: 最大迭代次数。
    """

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        tools: List[BaseTool] | None = None,
        verbose: bool = False,
        tool_parser: str | BaseParser = 'text',
        save_path: Optional[str] = None,
        max_refinement_depth: int = 20
    ):
        """初始化 Agent。

        Args:
            llm_provider: LLM 提供商名称（如 "openai", "siliconflow"）。
            llm_model: 模型名称（如 "gpt-4o-mini"）。
            tools: 可用工具列表。None 表示使用全部工具。
            verbose: 是否启用详细日志（DEBUG 级别）。
            tool_parser: 工具解析器，可以是字符串（'text', 'json'）或 BaseParser 实例。
            save_path: 日志文件保存路径。None 表示不保存到文件。
            max_refinement_depth: 最大迭代次数。
        """
        # 配置日志：如果用户尚未配置，则根据 verbose 和 save_path 自动配置
        setup_logging(info_level='debug' if verbose else 'info', save_path=save_path, force=False)

        # 参数
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.parser = None
        self.max_refinement_depth = max_refinement_depth

        # 关键组件
        self.buffer = []
        self.tool_cls_list = BaseTool.load_tool_classes(tools)
        self.tools = None # 延迟实例化, 因为需要 content 上下文
        self.llm_api = None # 延迟实例化, 因为需要 tools 工具列表

        # 附属组件
        self.named_timer = NamedTimer() # 用时统计
        self.token_counter = ParallelTimer(unit='token') # token 统计
        self.money_counter = ParallelTimer(unit='$') # 费用统计
        self.tool_call_counter = ParallelTimer(unit='call') # 工具调用统计
        self.save_path = save_path

        _logger.info(f"Initialized {self.__class__.__name__}")

    def fit( # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
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
        if not isinstance(y, dict):
            y = {"target": y}

        ## 实例化工具和 LLM API
        tool_context = { "data": X | y, "target": next(iter(y)) }
        self.tools = [tool_cls(**tool_context) for tool_cls in self.tool_cls_list]
        self.parser = BaseParser.create(self.tool_parser, tool_list=self.tools)
        self.llm_api = LLMAPI.create(
            self.llm_provider,
            self.llm_model,
            tool_list=self.tools,
            tool_parser=self.tool_parser,
        )
        if self.tool_parser == 'openai':
            _logger.note(
                f"Using OpenAI function calling as tool parser. Tools will be described in the system prompt as follows:\n"
                f"{json.dumps(self.llm_api.tool_description_json, indent=2)}"
            )
        else:
            _logger.note(
                f"Using {self.tool_parser} as tool parser. Tools will be described in the system prompt as follows:\n"
                f"{self.llm_api.tool_description_text}"
            )
                

        ## 初始化 Buffer
        self.buffer.clear()
        # 构建 system prompt - 告知 LLM 它的角色和目标
        self.buffer.append({
            "role": "system",
            "content": """You are a Symbolic Regression Agent. Your goal is to discover mathematical formulas that explain the relationship between feature variables and the target variable."""
        })
        # 构建 user prompt - 告知具体问题和数据信息
        self.buffer.append({
            "role": "user",
            "content": (
                f"{problem_description}\n\n"
                f"- Feature names: {list(X.keys())}\n"
                f"- Target name: {next(iter(y))}\n"
                f"Please start by analyzing the data to understand the relationship between features and target."
            )
        })

        ## 开始迭代
        best_record = None
        for L in range(1, self.max_refinement_depth + 1):
            _logger.info(f"Start Refinement Step {L}/{self.max_refinement_depth}")

            # Step 1: 根据 Buffer 创建 Prompt
            prompt = self.buffer
            _logger.info(f"Built prompt with {len(prompt)} messages")
            log = []
            for msg in prompt:
                role = tag2ansi(f"[red bold][{msg['role']}]: [reset]")
                content = render_markdown(msg['content'] or "(empty)")
                if 'tool_calls' in msg:
                    content += "\n".join(f"- {tool_call['function']['name']}({tool_call['function']['arguments']})" for tool_call in msg['tool_calls'])
                log.append(f"{role}{content}")
            _logger.debug(f"Messages:\n" + '\n---\n'.join(log))

            # Step 2: 请求 LLM 得到 Content 和 Tool Calls
            for content, tool_calls, message in (llm_result := self.llm_api(prompt, n=1)):
                pass
            self.record_llm_result(llm_result)
            _logger.info(f"LLM response content:\n{render_markdown(content or "(empty)")}")
            _logger.info(f"LLM tool calls: {tool_calls or "(None)"}")

            # Step 3: 执行 Tool Calls 得到 Result
            results = self.execute_action(tool_calls)
            _logger.info(f"Action result: {results}")

            # Step 4: 基于 Response Content、Tool Calls 和 Results 更新 Buffer
            self.buffer.append(message)
            self.buffer.extend(self.parser.format_tool_result_messages(tool_calls, results))

            # 更新最优结果
            for act, res in zip(tool_calls, results):
                if res is None:
                    continue
                result_dict = res.result if isinstance(res, ToolCallResult) else res
                if not isinstance(result_dict, dict):
                    continue
                metrics = result_dict.get('metrics') or {}
                mse = result_dict.get('mse', metrics.get('mse'))
                if mse is not None and (best_record is None or mse < best_record['score']):
                    best_formula = result_dict.get('formula') or act.params.get('eq')
                    best_record = {
                        "formula": best_formula,
                        "score": mse,
                        "mse": mse,
                        "rmse": result_dict.get('rmse', metrics.get('rmse')),
                        "mae": result_dict.get('mae', metrics.get('mae')),
                        "r2": result_dict.get('r2', metrics.get('r2')),
                    }

            # 统计本轮工具调用次数
            tmp = defaultdict(int)
            for tool_call in tool_calls:
                tmp[tool_call.name] += 1
            tool_calls_str = ';'.join(f"{name}: {count} ({tmp[name]} new)" for name, count in self.tool_call_counter.named_count.items())

            # 打印本轮日志
            log = {
                "Best": f"{best_record['formula']} (MSE={best_record['score']:.6g})" if best_record else "None",
                "Tool Calls": tool_calls_str,
                "Speed": self.named_timer.to_str('pace', None, None),
                "Time Usage": self.named_timer.to_str('time', 'pace', 'by_time'),
                "Token Usage": self.token_counter.to_str('count', 'speed', 'by_count'),
                "Price Usage": self.money_counter.to_str('count', 'speed', 'by_count'),
            }
            msg = " | ".join(f"\033[4m{k}\033[0m={v}" for k, v in log.items())
            _logger.info(msg)

        # ========== 后处理阶段 ==========
        _logger.info(
            f"Fit completed. Best formula: {best_record['formula'] if best_record else None}, "
            f"Best score: {best_record['score'] if best_record else None}"
        )

        return {
            "best_formula": best_record['formula'] if best_record else None,
            "best_score": best_record['score'] if best_record else None,
            "iterations": L,
        }

    def record_llm_result(self, llm_result) -> Dict[str, Any] | None:
        """记录最近一次 LLM 请求的返回值和用量统计。"""
        usage = llm_result.returned['usage']
        for name, num in usage['token'].items():
            self.token_counter.add(name, num)
        for name, num in usage['price'].items():
            self.money_counter.add(name, num)
        if self.save_path is not None:
            with open(Path(self.save_path) / 'response.jsonl', 'a') as f:
                json.dump(llm_result.returned["responses"], f)

    def execute_action(self, actions: List[ToolCall]) -> List[ToolCallResult|None]:
        """执行 Action。

        Args:
            actions: 多个 (name, value) 元组构成的数组

        Returns:
            执行结果列表。对于不需要执行的 name（例如 think 或者 response）返回 None
        """
        results = []
        tool_map = {tool.metadata.name: tool for tool in self.tools}
        for tool_call in actions:
            if tool_call.name == 'think':
                results.append(None)
            elif tool_call.name == 'response':
                results.append(None)
            else:
                self.tool_call_counter.add(tool_call.name)
                if (tool := tool_map.get(tool_call.name)) is None:
                    results.append(ToolCallResult(
                        ok=False,
                        result={},
                        result_str=f'Unknown tool calling for "{tool_call.name}"',
                        meta_data={"tool": tool_call.name},
                    ))
                    continue
                if not isinstance(tool_call.params, dict):
                    result = ToolCallResult(
                        ok=False,
                        result={},
                        result_str=f"Invalid parameters for tool '{tool_call.name}': expected dict, got {type(tool_call.params).__name__}",
                        meta_data={"tool": tool_call.name},
                    )
                else:
                    result = tool(**tool_call.params)
                results.append(result)

        return results
