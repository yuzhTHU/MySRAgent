# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""基类 Agent 和 Buffer 的定义。

提供符号回归 Agent 的基础框架，包括主循环 Pipeline 和工具调用机制。
"""
from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from itertools import islice
from collections import defaultdict
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
        local_sample_size: int = 1,
        max_refinement_depth: int = 20,
        global_width: int = 1,
        max_restart_loop: int = 1,
        restart_top_k: int = 1,
    ):
        """初始化 Agent。

        Args:
            llm_provider: LLM 提供商名称（如 "openai", "siliconflow"）。
            llm_model: 模型名称（如 "gpt-4o-mini"）。
            tools: 可用工具列表。None 表示使用全部工具。
            verbose: 是否启用详细日志（DEBUG 级别）。
            tool_parser: 工具解析器，可以是字符串（'text', 'json'）或 BaseParser 实例。
            save_path: 日志文件保存路径。None 表示不保存到文件。
            local_sample_size: 每轮生成的候选解数量。
            max_refinement_depth: 最大迭代次数。
            global_width: 每个 restart turn 中独立对话分支数量。
            max_restart_loop: best-solution restarts 次数
            restart_top_k: 下一轮 restart prompt 中保留的历史最佳结果数量。
        """
        # 配置日志：如果用户尚未配置，则根据 verbose 和 save_path 自动配置
        setup_logging(info_level='debug' if verbose else 'info', save_path=save_path, force=False)

        # 参数
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.local_sample_size = local_sample_size
        self.max_refinement_depth = max_refinement_depth
        self.global_width = global_width
        self.max_restart_loop = max_restart_loop
        self.restart_top_k = restart_top_k

        # 关键组件
        self.tool_cls_list = BaseTool.load_tool_classes(tools)
        self.tools = None # 延迟实例化, 因为需要 content 上下文
        self.parser = None # 延迟实例化, 因为需要 tools 工具列表
        self.llm_api = None # 延迟实例化, 因为需要 tools 工具列表

        # 附属组件
        self.named_timer = NamedTimer() # 用时统计
        self.token_counter = ParallelTimer(unit='token') # token 统计
        self.money_counter = ParallelTimer(unit='$') # 费用统计
        self.tools_counter = ParallelTimer(unit='call') # 工具调用统计
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
                
        ## 开始迭代
        best_record = None
        for R in range(1, self.max_restart_loop + 1):  # R 次 best-solution restart
            _logger.info(f"Start Restart Loop {R}/{self.max_restart_loop}")

            # 用平凡结果或者历史最佳结果构建新的 initial prompt
            initial_prompt = self.build_initial_prompt(problem_description, X, y, best_record)

            for C in range(1, self.global_width + 1):  # C 次独立重复对话
                _logger.info(f"Start Global Branch {C}/{self.global_width}")
            
                # 用 initial prompt 初始化 buffer
                buffer = deepcopy(initial_prompt)

                for L in range(1, self.max_refinement_depth + 1):  # L 轮对话迭代
                    _logger.info(f"Start Refinement Step {L}/{self.max_refinement_depth}")

                    # Step 1: 根据 Buffer 创建 Prompt
                    prompt = self.build_prompt(buffer)

                    # Step 2: 请求 LLM 得到 (Content, Tool Calls, Message) 元组
                    response_list = self.request_llm(prompt)

                    # Step 3: 执行 Tool Calls 得到 Results
                    results_list = self.get_results(response_list)

                    # Step 4: 基于 Response Content, Tool Calls, Messages 和 Results 更新 Buffer
                    buffer = self.update_buffer(buffer, response_list, results_list)

                    # Step 5: 更新最优结果
                    best_record = self.update_best(best_record, response_list, results_list)

                    # Step 6: 打印本轮日志
                    self.log_info(response_list, best_record)

        self.log_info([], best_record)
        return {f'best_{k}': v for k, v in best_record.items()}

    def build_initial_prompt(self, problem_description, X, y, best_record):
        """根据历史最佳结果构建新的 initial prompt。"""
        initial_prompt = []
        # 构建 system prompt - 告知 LLM 它的角色和目标
        initial_prompt.append({
            "role": "system",
            "content": """You are a Symbolic Regression Agent. Your goal is to discover mathematical formulas that explain the relationship between feature variables and the target variable."""
        })
        # 构建 user prompt - 告知具体问题和数据信息
        initial_prompt.append({
            "role": "user",
            "content": (
                f"{problem_description}\n\n"
                f"- Feature names: {list(X.keys())}\n"
                f"- Target name: {next(iter(y))}\n"
                f"Please start by analyzing the data to understand the relationship between features and target."
            )
        })
        return initial_prompt

    def build_prompt(self, buffer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据 Buffer 构建 LLM Prompt。"""
        prompt = buffer
        _logger.info(f"Built prompt with {len(prompt)} messages.")
        logs = []
        for msg in prompt:
            msg = msg.copy()
            order = ['role', 'tool_call_id', 'reasoning', 'content', 'tool_call']
            order = [k for k in order if k in msg] + [k for k in msg if k not in order]
            msg = { k: msg[k] for k in order }

            log = ''
            log += tag2ansi(f"[red bold][{msg.pop('role')}][reset]")
            for k, v in msg.items():
                if k == 'content':
                    v = render_markdown(v or "(empty)")
                v = str(v).strip()
                if '\n' in v:
                    v = '\n        '.join(['', *v.splitlines()])
                log += tag2ansi(f"\n    [blue]{k}[reset]") + '=' + v
            logs.append(log)
        _logger.debug(f"Messages:\n" + '\n---\n'.join(logs))
        return prompt
    
    def request_llm(self, prompt: List[Dict[str, Any]]):
        """请求 LLM 得到 Content 和 Tool Calls。"""
        response_list = []
        llm_result = self.llm_api(prompt, n=self.local_sample_size)
        for K, (content, tool_calls, message) in enumerate(llm_result): # K 次重复采样
            response_list.append((content, tool_calls, message))
            content_for_log = render_markdown(content or "(empty)").strip()
            content_for_log = '\n        '.join(['', *content_for_log.splitlines()]) if '\n' in content_for_log else content_for_log
            _logger.info(
                f"LLM branch: {len(response_list) + 1}/{self.local_sample_size}\n"
                f"LLM response content: {content_for_log}\n"
                f"LLM tool calls: {tool_calls or "(None)"}"
            )
        self.record_llm_result(llm_result)
        return response_list
    
    def get_results(self, response_list):
        """执行 Tool Calls 得到 Results。"""
        all_tool_calls = []
        num_tool_calls = []
        for _, tool_calls, _ in response_list:
            all_tool_calls.extend(tool_calls or [])
            num_tool_calls.append(len(tool_calls or []))
        all_results = self.execute_action(all_tool_calls)
        results_iter = iter(all_results)
        results_list = [list(islice(results_iter, l)) for l in num_tool_calls]
        _logger.info(f"Action result: {all_results}")
        return results_list
    
    def update_buffer(self, buffer, response_list, results_list):
        """根据 LLM Response 和 Tool Results 更新 Buffer。"""
        selected_idx = 0
        selected_mse = float('inf')
        for results_idx, results in enumerate(results_list):
            for result in results:
                if result.get('formula') is not None and result['metrics']['mse'] < selected_mse:
                    selected_idx = results_idx
                    selected_mse = result['metrics']['mse']
        content, tool_calls, message = response_list[selected_idx]
        results = results_list[selected_idx]
        content_parts = [content]
        tool_calls = list(tool_calls or [])
        results = list(results)
        message_tool_calls = message.get('tool_calls')
        for results_idx, ((extra_content, extra_tool_calls, _), extra_results) in enumerate(zip(response_list, results_list)):
            if results_idx == selected_idx:
                continue
            extra_content_added = False
            for extra_tool_call, extra_result in zip(extra_tool_calls or [], extra_results):
                if extra_result.get('formula') is not None:
                    tool_calls.append(extra_tool_call)
                    results.append(extra_result)
                    if message_tool_calls is not None and extra_tool_call.raw is not None:
                        message_tool_calls.append(extra_tool_call.raw)
                    if not extra_content_added:
                        content_parts.append(extra_content)
                        extra_content_added = True
        content = '\n\n'.join(part for part in content_parts if part)
        message['content'] = content
        _logger.info(f"Selected LLM branch: {selected_idx + 1}/{len(response_list)}")
        buffer.append(message)
        buffer.extend(self.parser.format_tool_result_messages(tool_calls, results))
        return buffer

    def update_best(self, best_record, response_list, results_list):
        """根据 LLM Response 和 Tool Results 更新最优结果。"""
        for idx in range(len(response_list)):
            for act, res in zip(response_list[idx][1], results_list[idx]):
                if res is None:
                    continue
                elif not isinstance(res.result, dict):
                    continue
                elif 'metrics' not in res.result:
                    continue
                elif not isinstance(res.result['metrics'], dict):
                    continue
                elif 'mse' not in res.result['metrics']:
                    continue
                elif (mse := res.result['metrics']['mse']) is None:
                    continue
                elif best_record is not None and mse >= best_record['mse']:
                    continue
                else:
                    best_record = {
                        "formula": res.result.get('formula') or act.params.get('eq'),
                        "mse": mse,
                        "rmse": res.result['metrics'].get('rmse'),
                        "mae": res.result['metrics'].get('mae'),
                        "r2": res.result['metrics'].get('r2'),
                    }
        return best_record
    
    def log_info(self, response_list, best_record):
        """打印本轮日志, response_list 是用来统计本轮新增工具调用次数的。"""
        new_count = defaultdict(int)
        for _, tool_calls, _ in response_list:
            for tool_call in tool_calls or []:
                new_count[tool_call.name] += 1
        tool_calls_str = ', '.join(
            f"{name}: {count} ({new_count[name]} new)" 
            for name, count in self.tools_counter.named_count.items()
        )
        log = {
            "Best": f"{best_record['formula']} (MSE={best_record['mse']:.6g})" if best_record else "None",
            "Tool Calls": tool_calls_str,
            "Speed": self.named_timer.to_str('pace', None, None),
            "Time Usage": self.named_timer.to_str('time', 'pace', 'by_time'),
            "Token Usage": self.token_counter.to_str('count', 'speed', 'by_count'),
            "Price Usage": self.money_counter.to_str('count', 'speed', 'by_count'),
        }
        msg = "[gray] | [reset]".join(f"[blue]{k}[reset]={v}" for k, v in log.items())
        _logger.info(tag2ansi(msg))

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
                f.write('\n')

    def execute_action(self, actions: List[ToolCall]) -> List[ToolCallResult|None]:
        """执行 Action。"""
        results = []
        tool_map = {tool.metadata.name: tool for tool in self.tools}
        for tool_call in actions:
            if (tool := tool_map.get(tool_call.name)) is None:
                _logger.trace(f'Unknown tool call: {tool_call.name}. Skipping execution.')
                result = ToolCallResult(
                    ok=False,
                    result={},
                    result_str=f'Unknown tool calling for "{tool_call.name}"',
                    meta_data={"tool": tool_call.name},
                )
            else:
                result = tool(**tool_call.params)
                self.tools_counter.add(tool_call.name)
            results.append(result)
        return results
