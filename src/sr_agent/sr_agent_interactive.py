# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""人机协同符号回归 Agent。

继承自 SRAgent，以 L（对话轮次）为搜索主体，支持人类实时干预和工作区文件操作。
"""
from __future__ import annotations
import json
import heapq
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from .tools import BaseTool
from .sr_agent import SRAgent
from .parser import BaseParser
from .api.llm_api import LLMAPI
from .utils import tag2ansi, render_markdown
from .tools.workspace_shell import Workspace
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(f'sr_agent.{__name__}')


class SRAgentInteractive(SRAgent):
    """人机协同符号回归 Agent。

    以 L（对话轮次）为搜索主体，默认 R=C=K=1，退化为纯对话式 Agent。
    支持工作区文件操作和人类实时反馈。
    """

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        tools: List[BaseTool] | None = None,
        verbose: bool = False,
        tool_parser: str | BaseParser = 'openai',
        save_path: Optional[str] = None,
        local_sample_size: int = 1,
        max_refinement_depth: int = 50,
        global_width: int = 1,
        max_restart_loop: int = 1,
        restart_top_k: int = 1,
        max_workers: int = 0,
        use_workspace: bool = False,
        workspace_files: List[str | Path] | None = None,
        human_input_callback: Optional[Callable[[str], str]] = None,
    ):
        """初始化 SRAgentInteractive。

        Args:
            llm_provider: LLM 提供商名称。
            llm_model: 模型名称。
            tools: 可用工具名列表。None 表示使用默认工具集（全部工具减去 code_executor）。
            verbose: 是否启用详细日志。
            tool_parser: 工具解析器类型。
            save_path: 日志保存路径。
            local_sample_size: 每轮 LLM 采样数量（K）。
            max_refinement_depth: 最大对话轮次（L），也是搜索深度上限。
            global_width: 独立分支数量（C）。
            max_restart_loop: 重启次数（R）。
            restart_top_k: 重启时注入历史最佳结果数量。
            max_workers: 并行工作进程数（0 表示不并行）。
            use_workspace: 是否使用工作区。
            workspace_files: 初始化到工作区的文件/目录路径列表。
            human_input_callback: 人类输入回调函数。默认 None 时使用 input()。
        """
        if use_workspace:
            excluded_tools = {"code_executor"}
        else:
            excluded_tools = {"workspace_code_executor", "workspace_shell"}
        self.excluded_tools = excluded_tools

        super().__init__(
            llm_provider=llm_provider,
            llm_model=llm_model,
            tools=tools,
            verbose=verbose,
            tool_parser=tool_parser,
            save_path=save_path,
            local_sample_size=local_sample_size,
            max_refinement_depth=max_refinement_depth,
            global_width=global_width,
            max_restart_loop=max_restart_loop,
            restart_top_k=restart_top_k,
            max_workers=max_workers,
        )

        # 工作区
        self.use_workspace = use_workspace
        self.workspace_files = workspace_files

        # 交互界面
        self.human_input_callback = human_input_callback or self._default_human_input

    def fit( # 这个函数已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
        self,
        X: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray] | np.ndarray,
        problem_description: str,
    ) -> Dict[str, Any]:
        """执行交互式符号回归任务的主入口。

        结构与 SRAgent.fit 类似，保留 R-C-L-K 层级（默认 R=C=K=1），
        以 L 为搜索主体。

        Args:
            X: 输入特征字典。
            y: 目标变量（numpy 数组或字典）。
            problem_description: 问题描述。

        Returns:
            包含最终结果的字典。
        """
        if not isinstance(y, dict):
            y = {"target": y}

        with Workspace(self.workspace_files, self.save_path) as workspace:
            _logger.note(f"Workspace initialized at: {workspace.path}")

            ## 实例化工具和 LLM API
            tool_context = {
                "data": X | y,
                "target": next(iter(y)),
                "workspace": workspace if self.use_workspace else None,
                "workspace_dir": str(workspace.path) if self.use_workspace else None,
                "human_input_callback": self.human_input_callback,
            }
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
                    restart_records = heapq.nsmallest(self.restart_top_k, topk_records)
                    initial_prompt = self.build_initial_prompt(problem_description, X, y, restart_records)
                    initial_node_parents = {record["node_id"]: 'restart_parent' for _, _, record in restart_records}
                    self.named_timer.add('build_initial_prompt')

                    for C in range(1, self.global_width + 1):  # C 次独立重复对话
                        _logger.info(f"(R={R}/{self.max_restart_loop}) × Global Branch (C={C}/{self.global_width})")

                        # 用 initial prompt 初始化 buffer, node_parent 记录当前 buffer 对应的 parent node_id list.
                        buffer = deepcopy(initial_prompt)
                        node_parents = deepcopy(initial_node_parents)
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
                            response_list, usage = self.request_llm(prompt, R=R, L=L, C=C)
                            self.named_timer.add('request_llm')

                            # Step 3: 执行 Tool Calls 得到 Results
                            results_list = self.get_results(response_list, R=R, L=L, C=C)
                            self.named_timer.add('get_results')

                            # Step 4: 基于 Response Content, Tool Calls, Messages 和 Results 更新 Buffer
                            buffer, node_parents = self.update_buffer(
                                buffer, response_list, results_list,
                                node_parents, prompt, usage, R, L, C,
                            )
                            self.named_timer.add('update_buffer')

                            # Step 5: 更新 top-k 最优结果
                            topk_records = self.update_topk(topk_records, response_list, results_list, R=R, L=L, C=C)
                            self.named_timer.add('update_topk')

                            # Step 6: 打印本轮日志
                            self.log_info(response_list, topk_records, R=R, L=L, C=C)
                            self.named_timer.add('log_info')
                            self.total_timer.add()

                            if topk_records and topk_records[0][-1]['mse'] == 0.0 and 'CONGRATULATIONS_FLAG' not in locals():
                                CONGRATULATIONS_FLAG = True
                                buffer.append({
                                    "role": "user",
                                    "content": (
                                        "Congratulations! You've found a formula with MSE=0. "
                                        "Please conclude the search and call ask_human with a summary of your discovery and the final formula."
                                    ),
                                })

                _logger.note("Finished all iterations. Returning best result.")
                best_record = topk_records[0][-1] if topk_records else {}
                return {f'best_{k}': v for k, v in best_record.items()} | {'status': 'completed', 'progress': self.format_progress(R, L, C)}

            except KeyboardInterrupt as e:
                best_record = topk_records[0][-1] if topk_records else {}
                e.partial_result = {f'best_{k}': v for k, v in best_record.items()} | {'status': 'interrupted', 'progress': self.format_progress(R, L, C)}
                raise

            except Exception as e:
                best_record = topk_records[0][-1] if topk_records else {}
                e.partial_result = {f'best_{k}': v for k, v in best_record.items()} | {'status': 'failed', 'progress': self.format_progress(R, L, C)}
                raise

    # 对外接口别名
    run = fit

    def build_initial_prompt(self, problem_description, X, y, restart_records):
        """构建面向交互式探索的 initial prompt。

        当 topk_record 非空时（即 R > 1 的重启轮次），会将之前探索过的最优公式
        及其指标作为上下文注入 prompt，并设置一个更严格的 MSE 目标，引导 LLM
        在之前最优解的基础上进一步优化（参考 SR-Scientist 的多轮策略）。
        """
        initial_prompt = []

        # 根据是否有历史最优结果来动态设置 MSE 目标
        if not restart_records:
            mse_goal = "You should try to find a simple formula that fits the data with an MSE of EXACTLY 0."
        elif (best_mse := restart_records[0][-1]['mse']) > 0:
            target_mse = best_mse * 0.1
            mse_goal = f"Your target is to find a formula with MSE < {target_mse:.6g} (10x better than the previous best MSE of {best_mse:.6g})."
        else:
            mse_goal = "The previous round already achieved MSE = 0. Try to find a simpler formula."

        # 构建工作区信息
        workspace_info = (
            "\n\nYou have access to a workspace directory containing data files. "
            "Use the workspace_shell tool to explore files (ls, cat, head, etc.) "
            "and run Python scripts. Use ask_human to pause and ask for guidance "
            "when you are uncertain about the direction."
        ) if self.use_workspace else ""

        # 构建 system prompt
        initial_prompt.append({
            "role": "system",
            "content": (
                "You are a Symbolic Regression Agent working with a human researcher. "
                "Your goal is to discover simple, interpretable mathematical formulas that explain "
                "the relationship between feature variables and the target variable.\n\n"
                "Guidelines:\n"
                "- Explore data thoroughly before proposing formulas.\n"
                "- Prefer simple, interpretable expressions over complex ones.\n"
                "- Use ask_human when you need guidance, are stuck, or want to report progress.\n"
                f"- {mse_goal}"
                f"{workspace_info}"
            ),
        })

        # 构建 user prompt - 告知具体问题和数据信息
        user_content = (
            f"{problem_description}\n\n"
            f"- Feature names: {list(X.keys())}\n"
            f"- Target name: {next(iter(y))}\n"
        )

        # 如果有历史最优结果，注入作为参考上下文
        if restart_records:
            user_content += (
                "\n--- Previously Explored Formulas (from best to worst) ---\n"
                "Use these as inspiration. Try to improve upon them or find simpler alternatives.\n\n"
            )
            for priority, sequence, record in restart_records:
                formula = record.get('formula', 'N/A')
                mse = record.get('mse', float('inf'))
                r2 = record.get('r2', None)
                r2_str = f", R²={r2:.6g}" if r2 is not None else ""
                user_content += f"  • Formula: {formula}\n    MSE={mse:.6g}{r2_str}\n\n"
            user_content += "---\n\n"
            user_content += (
                "Based on the above results, analyze why the previous best formulas may not be perfect, "
                "and try a different approach or structure to achieve a lower MSE."
            )
        else:
            user_content += "Please start by analyzing the data to understand the relationship between features and target."

        initial_prompt.append({
            "role": "user", 
            "content": user_content
        })
        return initial_prompt

    def execute_action_parallel(self, actions, max_workers: int):
        raise NotImplementedError(
            "Parallel execution is not supported in interactive mode, "
            "since tools like ask_human and workspace_shell cannot guarantee read-only access. "
            "Please set max_workers=0 to disable parallel execution when using interactive tools."
        )

    @staticmethod
    def _default_human_input(message: str) -> str:
        import re
        from prompt_toolkit import prompt
        from prompt_toolkit.patch_stdout import patch_stdout

        _SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
        """默认的人类输入回调：打印消息并等待 input()。"""
        print(tag2ansi(f"\n[gray]{'=' * 60}[reset]"))
        print(tag2ansi("[red bold][Agent asks for guidance][reset]"))
        print(tag2ansi(f"[gray]{'-' * 60}[reset]"))
        print(tag2ansi(render_markdown(message)))
        print(tag2ansi(f"[gray]{'=' * 60}[reset]"))
        with patch_stdout():
            response = prompt("Your response (press Enter to let agent continue): ")
        response = _SURROGATE_RE.sub("", response.strip() or "(No input)")
        return response
