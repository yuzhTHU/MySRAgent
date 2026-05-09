# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""基类 Agent 和 Buffer 的定义。

提供符号回归 Agent 的基础框架，包括主循环 Pipeline 和工具调用机制。
"""
from __future__ import annotations
import json
import heapq
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from itertools import islice
from collections import defaultdict
from typing import Any, Dict, List, Optional
from joblib import Parallel, delayed
from .api.llm_api import LLMAPI
from .parser import BaseParser
from .api.core import ToolCall
from .tools import BaseTool, ToolCallResult
from .utils import FactoryMixin, ParallelTimer, NamedTimer, Timer
from .utils.logger import setup_logging
from .utils import render_python, render_markdown, tag2ansi

_logger = logging.getLogger(f'sr_agent.{__name__}')

class FitEarlyStop(Exception):
    pass


def _execute_tool_call_in_subprocess(tool: BaseTool, tool_call: ToolCall) -> ToolCallResult:
    """Execute one tool call in a worker process."""
    return tool(**tool_call.params)


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
        max_workers: int = 0,
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
            max_workers: 并行执行工具调用的最大工作进程数。0 表示不使用并行。
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
        self.max_workers = max_workers

        # 关键组件
        self.tool_cls_list = BaseTool.load_tool_classes(tools)
        self.tools = None # 延迟实例化, 因为需要 content 上下文
        self.parser = None # 延迟实例化, 因为需要 tools 工具列表
        self.llm_api = None # 延迟实例化, 因为需要 tools 工具列表

        # 附属组件
        self.total_timer = Timer() # 总用时统计
        self.named_timer = NamedTimer() # 细粒度用时统计
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
            _logger.debug(
                f"Using OpenAI function calling as tool parser. Tools will be described in the system prompt as follows:\n"
                f"{json.dumps(self.llm_api.tool_description_json, indent=2)}"
            )
        else:
            _logger.debug(
                f"Using {self.tool_parser} as tool parser. Tools will be described in the system prompt as follows:\n"
                f"{self.llm_api.tool_description_text}"
            )
                
        ## 开始迭代
        topk_records = []
        R = C = L = None
        self.total_timer.clear(reset_last_add_time=True)
        self.named_timer.clear(reset_last_add_time=True)
        try:
            for R in range(1, self.max_restart_loop + 1):  # R 次 best-solution restart
                _logger.info(f"Start Restart Loop (R={R}/{self.max_restart_loop})")

                # 用平凡结果或者历史最佳结果构建新的 initial prompt
                initial_prompt = self.build_initial_prompt(problem_description, X, y, topk_records)
                self.named_timer.add('build_initial_prompt')

                for C in range(1, self.global_width + 1):  # C 次独立重复对话
                    _logger.info(f"(R={R}/{self.max_restart_loop}) × Global Branch (C={C}/{self.global_width})")
                
                    # 用 initial prompt 初始化 buffer
                    buffer = deepcopy(initial_prompt)
                    self.named_timer.add('init_buffer')

                    for L in range(1, self.max_refinement_depth + 1):  # L 轮对话迭代
                        _logger.info(
                            f"(R={R}/{self.max_restart_loop}) × (C={C}/{self.global_width}) × "
                            f"Refinement Step (L={L}/{self.max_refinement_depth})"
                        )

                        # Step 1: 根据 Buffer 创建 Prompt
                        prompt = self.build_prompt(buffer, R=R, L=L, C=C)
                        self.named_timer.add('build_prompt')

                        # Step 2: 请求 LLM 得到 (Content, Tool Calls, Message) 元组
                        response_list = self.request_llm(prompt, R=R, L=L, C=C)
                        self.named_timer.add('request_llm')

                        # Step 3: 执行 Tool Calls 得到 Results
                        results_list = self.get_results(response_list, R=R, L=L, C=C)
                        self.named_timer.add('get_results')

                        # Step 4: 基于 Response Content, Tool Calls, Messages 和 Results 更新 Buffer
                        buffer = self.update_buffer(buffer, response_list, results_list, R=R, L=L, C=C)
                        self.named_timer.add('update_buffer')

                        # Step 5: 更新 top-k 最优结果
                        topk_records = self.update_topk(topk_records, response_list, results_list, R=R, L=L, C=C)
                        self.named_timer.add('update_topk')

                        # Step 6: 打印本轮日志
                        self.log_info(response_list, topk_records, R=R, L=L, C=C)
                        self.named_timer.add('log_info')
                        self.total_timer.add()

                        if topk_records and topk_records[0][-1]['mse'] == 0.0:
                            raise FitEarlyStop()

            _logger.note(f"Finished all iterations. Returning best result.")
            best_record = topk_records[0][-1] if topk_records else {}
            return {f'best_{k}': v for k, v in best_record.items()} | {'status': 'completed', 'progress': self.format_progress(R, L, C)}
        
        except FitEarlyStop as e:
            _logger.note(f"Early stopping triggered by perfect solution. Returning best result.")
            best_record = topk_records[0][-1] if topk_records else {}
            return {f'best_{k}': v for k, v in best_record.items()} | {'status': 'early_stopped', 'progress': self.format_progress(R, L, C)}

        except KeyboardInterrupt as e:
            best_record = topk_records[0][-1] if topk_records else {}
            e.partial_result = {f'best_{k}': v for k, v in best_record.items()} | {'status': 'interrupted', 'progress': self.format_progress(R, L, C)}
            raise

        except Exception as e:
            best_record = topk_records[0][-1] if topk_records else {}
            e.partial_result = {f'best_{k}': v for k, v in best_record.items()} | {'status': 'failed', 'progress': self.format_progress(R, L, C)}
            raise

    def build_initial_prompt(self, problem_description, X, y, topk_record):
        """根据历史最佳结果构建新的 initial prompt。"""
        topk_record = heapq.nsmallest(self.restart_top_k, topk_record)
        initial_prompt = []
        # 构建 system prompt - 告知 LLM 它的角色和目标
        initial_prompt.append({
            "role": "system",
            "content": """You are a Symbolic Regression Agent. Your goal is to discover mathematical formulas that explain the relationship between feature variables and the target variable. DO NOT satisfied with a accurate but complex formula, you should try to find a simple formula that fit the data with an MSE of EXACTLY 0."""
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

    def build_prompt(self, buffer: List[Dict[str, Any]], R: int, L: int, C: int) -> List[Dict[str, Any]]:
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
    
    def request_llm(self, prompt: List[Dict[str, Any]], R: int, L: int, C: int):
        """请求 LLM 得到 Content 和 Tool Calls。"""
        response_list = []
        llm_result = self.llm_api(prompt, n=self.local_sample_size)
        for K, (content, tool_calls, message) in enumerate(llm_result, 1): # K 次重复采样
            response_list.append((content, tool_calls, message))
            content_for_log = render_markdown(content or "(empty)").strip()
            tool_calls_for_log = '\n'.join(str(tool_call) for tool_call in tool_calls)
            content_for_log = '\n        '.join(['', *content_for_log.splitlines()]) if '\n' in content_for_log else content_for_log
            tool_calls_for_log = '\n        '.join(['', *tool_calls_for_log.splitlines()]) if '\n' in tool_calls_for_log else tool_calls_for_log
            _logger.info(
                f"(R={R}/{self.max_restart_loop}) × (C={C}/{self.global_width}) × (L={L}/{self.max_refinement_depth}) × Local Sample (K={K}/{self.local_sample_size})\n"
                f"LLM response content: {content_for_log}\n"
                f"LLM tool calls: ({len(tool_calls)} tool calls)"
            )
            _logger.debug(tool_calls_for_log)
        self.record_llm_result(llm_result, R=R, L=L, C=C)
        return response_list
    
    def get_results(self, response_list, R: int, L: int, C: int):
        """执行 Tool Calls 得到 Results。"""
        # 合并 - 调用 - 分割
        all_tool_calls = []
        num_tool_calls = []
        for _, tool_calls, _ in response_list:
            all_tool_calls.extend(tool_calls)
            num_tool_calls.append(len(tool_calls))
        if self.max_workers and len(all_tool_calls) > 1:
            all_results = self.execute_action_parallel(all_tool_calls, max_workers=self.max_workers)
        else:
            all_results = self.execute_action(all_tool_calls)
        results_iter = iter(all_results)
        results_list = [list(islice(results_iter, l)) for l in num_tool_calls]
        # 打印工具调用结果
        all_results_for_log = '\n'.join(str(result) for result in all_results or [])
        all_results_for_log = '\n        '.join(['', *all_results_for_log.splitlines()]) if '\n' in all_results_for_log else all_results_for_log
        _logger.debug(f"Action result: {all_results_for_log}")
        # 记录工具调用结果
        if self.save_path is not None:
            with open(Path(self.save_path) / 'tool_calls.jsonl', 'a') as f:
                for tool_call, result in zip(all_tool_calls, all_results):
                    json.dump({
                        "progress": self.format_progress(R, L, C),
                        'name': tool_call.name, 
                        'params': tool_call.params,
                        "ok": result.ok, 
                        "result": result.result, 
                        "result_str": result.result_str, 
                        "meta_data": result.meta_data,
                    }, f)
                    f.write('\n')
        return results_list
    
    def update_buffer(self, buffer, response_list, results_list, R: int, L: int, C: int):
        """根据 LLM Response 和 Tool Results 更新 Buffer。"""
        # 如果没有成功的回复，跳过本轮更新
        if len(response_list) == 0:
            return buffer
        # 选择产生了最佳 mse 的 tool_call 所在的 response 分支
        selected_idx = 0
        selected_mse = float('inf')
        for results_idx, results in enumerate(results_list):
            for result in results:
                if (metrics := result.get('metrics')) is not None and metrics['mse'] < selected_mse:
                    selected_idx = results_idx
                    selected_mse = metrics['mse']
        _logger.info(f"Selected LLM branch: {selected_idx + 1}/{len(results_list)}")
        _, tool_calls, message = response_list[selected_idx]
        results = results_list[selected_idx]
        tool_calls = tool_calls.copy()
        message = deepcopy(message)
        results = results.copy()
        # 将其他 (tool_call, result) pairs 中不涉及 formula & metrics 的 pair 也加入 buffer, 以免丢失有用信息
        for idx, ((extra_content, extra_tool_calls, extra_message), extra_results) in enumerate(zip(response_list, results_list)):
            for tool_call, result in zip(extra_tool_calls, extra_results):
                # 只考虑不涉及 formula & metrics 的工具调用结果
                if idx == selected_idx or result.get('metrics') is not None:
                    continue
                tool_calls.append(tool_call)
                results.append(result)
                # 对于 openai parser, 将 extra_message['tool_calls'] 拼到 message 中
                if self.tool_parser == 'openai':
                    message['tool_calls'].append(tool_call.raw)
                # 对于 non-openai parser, 将 tool_calls 拼到 content 中
                else:
                    message['content'] += "\n\n" + tool_call.raw_str
        # 将 message 和 (tool_call, result) pairs 加入 buffer
        buffer.append(message)
        buffer.extend(self.parser.format_tool_result_messages(tool_calls, results))
        return buffer

    def update_topk(self, topk_records, response_list, results_list, R: int, L: int, C: int):
        """根据 LLM Response 和 Tool Results 更新 top-k 最优结果。"""
        for idx in range(len(response_list)):
            for act, res in zip(response_list[idx][1], results_list[idx]):
                if res.result.get('is_candidate'):
                    record = {
                        "formula": res.result.get('formula') or act.params.get('eq'),
                        "mse": res.result['metrics']['mse'],
                        "rmse": res.result['metrics'].get('rmse'),
                        "mae": res.result['metrics'].get('mae'),
                        "r2": res.result['metrics'].get('r2'),
                    }
                    priority = res.result['metrics']['mse'] # 按照 mse 排序 (越小越重要)
                    sequence = len(topk_records) # 相同 priority 时按照 sequence 排序 (越小越重要)
                    heapq.heappush(topk_records, (priority, sequence, record))
        return topk_records
    
    def log_info(self, response_list, topk_records, R: int, L: int, C: int):
        """打印本轮日志, response_list 是用来统计本轮新增工具调用次数的。"""
        new_count = defaultdict(int)
        for _, tool_calls, _ in response_list:
            for tool_call in tool_calls:
                new_count[tool_call.name] += 1
        tool_calls_str = ', '.join(
            f"{name}: {count} ({new_count[name]} new)" 
            for name, count in self.tools_counter.named_count.items()
        )
        best_record = topk_records[0][-1] if topk_records else {}
        log = {
            "Progress": self.format_progress(R, L, C),
            "Best": f"{best_record['formula']} (MSE={best_record['mse']:.6g})" if best_record else "None",
            "Tool Calls": tool_calls_str,
            "Speed": self.total_timer.to_str('pace'),
            "Time Usage": self.named_timer.to_str('time', 'pace', 'by_time'),
            "Token Usage": self.token_counter.to_str('count', 'speed', 'by_count'),
            "Price Usage": self.money_counter.to_str('count', 'speed', 'by_count'),
        }
        msg = "[gray] | [reset]".join(f"[blue]{k}[reset]={v}" for k, v in log.items())
        _logger.info(tag2ansi(msg))

    def record_llm_result(self, llm_result, R: int, L: int, C: int) -> Dict[str, Any] | None:
        """记录最近一次 LLM 请求的返回值和用量统计。"""
        usage = llm_result.returned['usage']
        for name, num in usage['token'].items():
            self.token_counter.add(name, num)
        for name, num in usage['price'].items():
            self.money_counter.add(name, num)
        if self.save_path is not None:
            with open(Path(self.save_path) / 'response.jsonl', 'a') as f:
                json.dump({
                    "responses": llm_result.returned["responses"],
                    "progress": self.format_progress(R, L, C),
                    "usage": usage,
                }, f)
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

    def execute_action_parallel(self, actions: List[ToolCall], max_workers: int) -> List[ToolCallResult]:
        """使用多进程并行执行 Action。"""
        tasks = []
        results = [None] * len(actions)
        tool_map = {tool.metadata.name: tool for tool in self.tools}
        for idx, tool_call in enumerate(actions):
            if (tool := tool_map.get(tool_call.name)) is None:
                _logger.trace(f'Unknown tool call: {tool_call.name}. Skipping execution.')
                results[idx] = ToolCallResult(
                    ok=False,
                    result={},
                    result_str=f'Unknown tool calling for "{tool_call.name}"',
                    meta_data={"tool": tool_call.name},
                )
            else:
                tasks.append((idx, delayed(_execute_tool_call_in_subprocess)(tool, tool_call)))
                self.tools_counter.add(tool_call.name)

        if tasks:
            workers = Parallel(n_jobs=max_workers, backend='loky')
            task_results = workers(task for _, task in tasks)
            for (idx, _), result in zip(tasks, task_results):
                results[idx] = result
        return results

    def format_progress(self, R: int, L: int, C: int):
        return f'(R={R}/{self.max_restart_loop}) × (C={C}/{self.global_width}) × (L={L}/{self.max_refinement_depth}) × (K={self.local_sample_size})'
