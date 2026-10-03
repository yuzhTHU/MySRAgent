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
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from ..api import BaseAPI
from ..tools import BaseTool
from ..parser import BaseParser
from ..skills import SkillManager
from ..runtime import ModelRouter
from ..utils import FactoryMixin, ParallelTimer, NamedTimer, Timer
from ..utils import format_pareto_front, render_markdown, tag2ansi, setup_logging
from ..core import CandidateRecord, ParentLink, SearchRunState, ToolCall, ToolCallResult

_logger = logging.getLogger(f'sr_harness.{__name__}')


def _execute_tool_call_in_subprocess(tool: BaseTool, tool_call: ToolCall) -> ToolCallResult:
    """Execute one tool call in a worker process."""
    return tool(**tool_call.params)


class SRAgent(FactoryMixin):
    """符号回归 Agent。

    提供完整的符号回归流程框架，包括数据预处理、Prompt 生成、LLM 请求、
    工具调用、Buffer 更新等。具体实现需继承此类并实现必要方法。

    Attributes:
        api: LLM API 实例。
        buffer: 对话历史 Buffer。
        max_refinement_depth: 最大迭代次数。
    """

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        tools: List[BaseTool] | None = None,
        verbose: bool = False,
        tool_parser: str | BaseParser = 'openai',
        save_path: Optional[str] = None,
        run_id: str | None = None,
        local_sample_size: int = 1,
        max_refinement_depth: int = 20,
        global_width: int = 1,
        max_restart_loop: int = 1,
        restart_top_k: int = 1,
        llm_max_tokens: int = 4096,
        max_workers: int = 0,
        validation_fraction: float = 0.2,
        split_by: str = "ood",
        split_random_state: int = 42,
        ranking_metric: str = "mse",
        larger_is_better: bool = False,
        force_initial_diagnostics: bool = False,
        auto_routing: bool = True,
        strong_llm_provider: str | None = None,
        strong_llm_model: str | None = None,
    ):
        """初始化 Agent。

        Args:
            llm_provider: LLM 提供商名称（如 "openai", "siliconflow"）。
            llm_model: 模型名称（如 "gpt-4o-mini"）。
            tools: 可用工具列表。None 表示使用全部工具。
            verbose: 是否启用详细日志（DEBUG 级别）。
            tool_parser: 工具解析器，可以是字符串（'text', 'json'）或 BaseParser 实例。
            save_path: 日志文件保存路径。None 表示不保存到文件。
            run_id: 本次运行的全局唯一标识。None 表示自动生成。
            local_sample_size: 每轮生成的候选解数量。
            max_refinement_depth: 最大迭代次数。
            global_width: 每个 restart turn 中独立对话分支数量。
            max_restart_loop: best-solution restarts 次数
            restart_top_k: 下一轮 restart prompt 中保留的历史最佳结果数量。
            llm_max_tokens: 每次 LLM 响应的最大 token 数。
            max_workers: 并行执行工具调用的最大工作进程数。0 表示不使用并行。
            validation_fraction: 验证集比例；验证集结果会展示给 Agent。设为 0 可关闭。
            split_by: 验证集划分方式。"random" 表示随机划分；"ood" 表示优先按 t、
                其次按 T 排序，并将取值较高的区间作为验证集。
            split_random_state: 数据划分的随机种子。
            ranking_metric: 候选公式排序所用的指标键；默认使用 mse。
            larger_is_better: 排序指标是否越大越好；默认按越小越好排序。
            force_initial_diagnostics: 是否在每个对话分支的 L=1 请求 LLM 前，强制执行
                statistics_analysis、relationship_analysis 和读取 discover-symbolic-laws skill。
            auto_routing: 是否根据任务复杂度在基础与强模型后端之间自动路由。
            strong_llm_provider: 复杂任务使用的后端；默认沿用 llm_provider。
            strong_llm_model: 复杂任务使用的模型。None 表示仅使用基础模型。
        """
        # 配置日志：如果用户尚未配置，则根据 verbose 和 save_path 自动配置
        log_path = Path(save_path) / "info.log" if save_path is not None else None
        setup_logging(
            info_level='debug' if verbose else 'info',
            save_path=log_path,
            force=False,
        )

        if not hasattr(self, "excluded_tools"):
            # ask_human and workspace_code_executor require interaction or
            # workspace permissions, so the non-interactive agent excludes them.
            self.excluded_tools = {"ask_human", "workspace_code_executor"}

        tool_cls_list = []
        for tool_cls in BaseTool.load_tool_classes(tools):
            # Custom tools are rediscovered and reloaded from this Agent's
            # SkillManager below. Exclude stale process-global class objects.
            if getattr(tool_cls, "source_path", None) is not None:
                continue
            if (name := tool_cls.metadata.name) in self.excluded_tools:
                _logger.info(f"Excluding tool {name} from the agent's toolset.")
            else:
                tool_cls_list.append(tool_cls)

        # 参数
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.local_sample_size = local_sample_size
        self.max_refinement_depth = max_refinement_depth
        self.global_width = global_width
        self.max_restart_loop = max_restart_loop
        self.restart_top_k = restart_top_k
        self.llm_max_tokens = llm_max_tokens
        self.max_workers = max_workers
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        if split_by not in {"random", "ood"}:
            raise ValueError("split_by must be either 'random' or 'ood'.")
        self.validation_fraction = validation_fraction
        self.split_by = split_by
        self.split_random_state = split_random_state
        self.ranking_metric = ranking_metric
        self.larger_is_better = larger_is_better
        self.force_initial_diagnostics = force_initial_diagnostics
        self.auto_routing = auto_routing
        self.strong_llm_provider = strong_llm_provider or llm_provider
        self.strong_llm_model = strong_llm_model
        self.model_router = ModelRouter(
            enabled=auto_routing,
            base_provider=llm_provider,
            base_model=llm_model,
            strong_provider=self.strong_llm_provider,
            strong_model=strong_llm_model,
        )
        self._task_route_score = 0
        self._task_route_reasons: list[str] = []
        self._strong_api = None
        self._last_model_route = None

        # 关键组件
        self.tool_cls_list = tool_cls_list
        self.skill_manager = SkillManager()
        self.skill_manager.register_tool_docs(self.tool_cls_list)
        self.tool_cls_list += BaseTool.discover_custom_tools(self.skill_manager)
        if self.force_initial_diagnostics:
            required_tools = {"statistics_analysis", "relationship_analysis", "read_skill"}
            enabled_tools = {tool_cls.metadata.name for tool_cls in self.tool_cls_list}
            if missing_tools := required_tools - enabled_tools:
                raise ValueError(
                    "force_initial_diagnostics requires these enabled tools: "
                    + ", ".join(sorted(missing_tools))
                )
            if "discover-symbolic-laws" not in self.skill_manager.load_skills():
                raise ValueError(
                    "force_initial_diagnostics requires the 'discover-symbolic-laws' skill."
                )
        self.tools = None # 延迟实例化, 因为需要 content 上下文
        self.parser = None # 延迟实例化, 因为需要 tools 工具列表
        self.api = None # 延迟实例化, 因为需要 tools 工具列表

        # 附属组件
        self.total_timer = Timer() # 总用时统计
        self.named_timer = NamedTimer() # 细粒度用时统计
        self.token_counter = ParallelTimer(unit='token') # token 统计
        self.money_counter = ParallelTimer(unit='$') # 费用统计
        self.tools_counter = ParallelTimer(unit='call') # 工具调用统计
        self.save_path = save_path
        self.run_state = SearchRunState(
            save_path=save_path,
            ranking_metric=ranking_metric,
            larger_is_better=larger_is_better,
            agent_metadata={
                "class": self.__class__.__name__,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "tool_parser": str(tool_parser),
                "local_sample_size": local_sample_size,
                "max_refinement_depth": max_refinement_depth,
                "global_width": global_width,
                "max_restart_loop": max_restart_loop,
                "restart_top_k": restart_top_k,
                "max_workers": max_workers,
            },
            run_id=run_id,
        )

        _logger.info(f"Initialized {self.__class__.__name__}")

    def run(self, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray] | np.ndarray, problem_description: str) -> Dict[str, Any]:
        """执行符号回归任务的主入口。

        负责数据划分以及工作区、工具、Parser 和 API 的初始化，随后调用
        search() 执行 R-C-L 搜索。

        Args:
            X: 输入特征字典，键为特征名，值为 numpy 数组。
            y: 目标变量 numpy 数组。
            problem_description: 问题描述字符串，告知 Agent 任务目标。

        Returns:
            包含本次运行状态、候选公式列表、Pareto front 候选下标和最佳候选下标的字典。
        """
        if not isinstance(y, dict):
            y = {"target": y}

        train_data, validation_data = self._split_data(X, y)

        tool_context = {
            "data": train_data,
            "target": next(iter(y)),
            "evaluation_data": validation_data,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_tokens": self.llm_max_tokens,
            "skill_manager": self.skill_manager,
        }
        with self.prepare_tool_context(tool_context) as tool_context:
            self.tools = [tool_cls(**tool_context) for tool_cls in self.tool_cls_list]
            self.parser = BaseParser.create(self.tool_parser, tool_list=self.tools)
            self.api = BaseAPI.create(
                self.llm_provider,
                model=self.llm_model,
                tool_list=self.tools,
                tool_parser_name=self.tool_parser,
            )
            if self.tool_parser == 'openai':
                description = json.dumps(self.api.tool_description_json, indent=2)
            else:
                description = self.api.tool_description_text
            _logger.debug(
                f"Using {self.tool_parser} as tool parser. "
                f"Tools will be described as follows:\n{description}"
            )
            try:
                return self.search(X, y, problem_description)
            except KeyboardInterrupt as error:
                coordinate = self.run_state.latest_coordinate
                error.partial_result = self.search_result(
                    "interrupted",
                    R=coordinate.R if coordinate else None,
                    C=coordinate.C if coordinate else None,
                    L=coordinate.L if coordinate else None,
                )
                raise
            except Exception as error:
                coordinate = self.run_state.latest_coordinate
                error.partial_result = self.search_result(
                    "failed",
                    R=coordinate.R if coordinate else None,
                    C=coordinate.C if coordinate else None,
                    L=coordinate.L if coordinate else None,
                )
                raise

    @contextmanager
    def prepare_tool_context(self, tool_context: Dict[str, Any]):
        """Prepare resources and context shared by initialized tools."""
        yield tool_context

    def search(self, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray], problem_description: str) -> Dict[str, Any]:
        """Run the R-C-L search after data, tools, parser, and API are initialized.

        Each refinement step builds the prompt, requests the model, executes tools,
        records the search node, collects candidates, updates the conversation, and
        evaluates the termination hook.
        """
        ## 开始迭代
        self.total_timer.clear(reset_last_add_time=True)
        self.named_timer.clear(reset_last_add_time=True)
        for R in range(1, self.max_restart_loop + 1):  # R 次 best-solution restart
            _logger.info(f"Start Restart Loop (R={R}/{self.max_restart_loop})")

            # 用平凡结果或者历史最佳结果构建新的 initial prompt
            restart_records = self.run_state.ranked_candidates()[:self.restart_top_k]
            initial_prompt = self.build_initial_prompt(problem_description, X, y, restart_records)
            initial_node_parents = {record.node_id: "restart_seed" for record in restart_records}
            self.named_timer.add("build_initial_prompt")
            for C in range(1, self.global_width + 1):  # C 次独立重复对话
                _logger.info(
                    f"(R={R}/{self.max_restart_loop}) × "
                    f"Global Branch (C={C}/{self.global_width})"
                )

                # 用 initial prompt 初始化 buffer，node_parents 记录当前 buffer 的父节点
                buffer = deepcopy(initial_prompt)
                node_parents = deepcopy(initial_node_parents)
                self.named_timer.add("init_buffer")

                for L in range(1, self.max_refinement_depth + 1):  # L 轮对话迭代
                    _logger.info(
                        f"(R={R}/{self.max_restart_loop}) × "
                        f"(C={C}/{self.global_width}) × "
                        f"Refinement Step (L={L}/{self.max_refinement_depth})"
                    )

                    # Step 1: 根据 Buffer 创建 Prompt
                    self.before_iteration(buffer, R=R, L=L, C=C)
                    prompt = self.build_prompt(buffer, R=R, L=L, C=C)
                    for tool in self.tools:
                        tool.context["messages"] = deepcopy(prompt)
                    self.named_timer.add("build_prompt")

                    # Step 2: 请求 LLM 得到 Content、Tool Calls 和 Message
                    response_list, usage = self.request_llm(prompt, R=R, L=L, C=C)
                    self.named_timer.add("request_llm")

                    # Step 3: 执行 Tool Calls 得到 Results
                    results_list = self.get_results(response_list, R=R, L=L, C=C)
                    self.named_timer.add("get_results")

                    # Step 4: 记录当前搜索节点
                    self.record_search_iteration(
                        response_list, results_list, node_parents, prompt, usage, R, L, C,
                    )
                    self.named_timer.add("record_search_iteration")

                    # Step 5: 验证并收集候选公式
                    self.collect_candidates(response_list, results_list, R=R, L=L, C=C)
                    self.named_timer.add("collect_candidates")

                    # Step 6: 更新对话 Buffer 和父节点关系
                    buffer, node_parents = self.update_buffer(
                        buffer, response_list, results_list, node_parents, R, L, C,
                    )
                    self.named_timer.add("update_buffer")

                    # Step 7: 打印本轮日志
                    self.log_info(response_list, R=R, L=L, C=C)
                    self.named_timer.add("log_info")
                    self.total_timer.add()

                    # Step 8: 判断是否终止当前搜索
                    status = self.handle_iteration_complete(buffer, R=R, L=L, C=C)
                    if status is not None:
                        _logger.note("Early stopping triggered. Returning best result.")
                        return self.search_result(status, R=R, L=L, C=C)

        _logger.note("Finished all iterations. Returning best result.")
        coordinate = self.run_state.latest_coordinate
        return self.search_result(
            "completed",
            R=coordinate.R if coordinate else None,
            C=coordinate.C if coordinate else None,
            L=coordinate.L if coordinate else None,
        )

    def _split_data(self, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray]):
        """Split aligned arrays into train and validation data mappings."""
        data = X | y
        arrays = {name: np.asarray(value) for name, value in data.items()}
        n_samples = len(arrays[next(iter(y))])
        n_validation = int(round(n_samples * self.validation_fraction))
        if self.validation_fraction == 0:
            n_train = n_samples # 当不使用 validation_fraction 时, 训练集就是验证集
            n_validation = n_samples
        elif n_validation == 0:
            n_validation = 1
            n_train = n_samples - n_validation
            _logger.warning(f"Validation fraction {self.validation_fraction} is too small for {n_samples} samples; using 1 validation sample.")
        elif n_validation >= n_samples:
            n_validation = n_samples - 1
            n_train = n_samples - n_validation
            _logger.warning(f"Validation fraction {self.validation_fraction} is too large for {n_samples} samples; using {n_samples - 1} validation samples.")
        else:
            n_train = n_samples - n_validation

        if self.split_by == "random":
            indices = np.random.default_rng(self.split_random_state).permutation(n_samples)
            train_indices = indices[:n_train]
            validation_indices = indices[-n_validation:]
        else:
            split_variable = "t" if "t" in X else "T" if "T" in X else list(X.keys())[0]
            split_values = arrays[split_variable]
            _logger.debug(f"Splitting data by '{split_variable}' values: {split_values}")
            indices = np.argsort(split_values, kind="stable")
            train_indices = indices[:n_train]
            validation_indices = indices[-n_validation:]

        if (network_data := "A" in arrays or "G" in arrays):
            temporal_names = {
                name for name, value in arrays.items()
                if name not in {"A", "G"} and value.ndim > 0 and len(value) == n_samples
            }
            if mismatched_targets := [name for name in y if name not in temporal_names]:
                raise ValueError(
                    f"Network-dynamics targets must use time as their first axis: {mismatched_targets}"
                )
        elif len(lengths := {len(value) for value in arrays.values()}) != 1:
            raise ValueError(
                f"All X and y arrays must have the same length, got lengths={sorted(lengths)}."
            )
        else:
            temporal_names = set(arrays)

        def select(selected):
            return {
                name: value[selected] if name in temporal_names else value
                for name, value in arrays.items()
            }

        return select(train_indices), select(validation_indices) if n_validation else {}

    def build_initial_prompt(self, problem_description, X, y, restart_records):
        """根据历史最佳结果构建新的 initial prompt。

        当 topk_record 非空时（即 R > 1 的重启轮次），会将之前探索过的最优公式
        及其指标作为上下文注入 prompt，并设置一个更严格的 MSE 目标，引导 LLM
        在之前最优解的基础上进一步优化（参考 SR-Scientist 的多轮策略）。
        """
        initial_prompt = []
        self._task_route_score, self._task_route_reasons = self.model_router.assess(
            problem_description,
            feature_count=len(X),
        )

        # 根据是否有历史最优结果来动态设置 MSE 目标
        if not restart_records:
            mse_goal = "You should try to find a simple formula that fits the data with an MSE of EXACTLY 0."
        elif (best_mse := restart_records[0].metric("mse", "train")) > 0:
            target_mse = best_mse * 0.1
            mse_goal = f"Your target is to find a formula with MSE < {target_mse:.6g} (10x better than the previous best MSE of {best_mse:.6g})."
        else:
            mse_goal = "The previous round already achieved MSE = 0. Try to find a simpler formula that also achieves MSE = 0."

        # 构建 system prompt
        initial_prompt.append({
            "role": "system",
            "content": (
                f"You are a Symbolic Regression Agent. Your goal is to discover mathematical formulas "
                f"that explain the relationship between feature variables and the target variable. "
                f"DO NOT be satisfied with an accurate but complex formula — prefer simple, interpretable expressions. "
                f"You have at most {self.max_refinement_depth} refinement rounds in each conversation branch. "
                f"Plan tool use within this budget: use early rounds for targeted exploration, keep concrete "
                f"candidate formulas as the budget shrinks, and avoid open-ended searches near the end. "
                f"If possible, try calling multiple tools in each round. "
                f"At the final refinement round (L={self.max_refinement_depth}), stop exploration and submit the best available "
                f"target formula using the most appropriate final-answer mechanism available; do not wait for another reminder after the final round. "
                f"Please start by analyzing the data to understand the relationship between features and target."
            )
        })

        # 构建 user prompt - 告知具体问题和数据信息
        user_content = (
            f"{problem_description}\n\n"
            f"- Feature names: {list(X.keys())}\n"
            f"- Target name: {next(iter(y))}\n"
        )

        # 如果有历史最优结果，注入作为参考上下文
        if restart_records:
            previous_formulas = []
            for idx, record in enumerate(restart_records):
                formula = record.formula
                result = record.details['data_split_results']
                previous_formulas.append(
                    f"{idx}. Formula: {formula}\n"
                    f"    (Train | Validation)\n"
                    f"    R2={result['train']['metrics']['r2']:.6g} | {result['validation']['metrics']['r2']:.6g}\n"
                    f"    MSE={result['train']['metrics']['mse']:.6g} | {result['validation']['metrics']['mse']:.6g}\n"
                )
            if (best_mse := restart_records[0].metric("mse", "train")) > 0:
                goal = f"find a formula with MSE < {best_mse * 0.1:.3g} (10x better than the previous best MSE)"
            else:
                goal = "find a simpler formula that also achieves MSE = 0"
            user_content += (
                f"\n---\n\n"
                f"Previously Explored Formulas (from best to worst):\n"
                f"{'\n'.join(previous_formulas)}\n\n"
                f"Use these as inspiration. Try to improve upon them or find simpler alternatives.\n"
                f"\n---\n\n"
                f"Based on the above results, {goal}."
            )
        else:
            user_content += (
                "You should try to find a simple formula that fits the data with an MSE of EXACTLY 0."
            )

        initial_prompt.append({
            "role": "user",
            "content": user_content
        })
        process_message = self.build_process_message(L=0)
        initial_prompt.append(process_message)
        return initial_prompt

    def build_prompt(self, buffer: List[Dict[str, Any]], R, L, C) -> List[Dict[str, Any]]:
        """根据 Buffer 构建 LLM Prompt。"""
        if self.force_initial_diagnostics and L == 1:
            # Persist the evidence in the branch buffer so later refinement
            # rounds retain the diagnostics and skill instructions.
            buffer.append(self.run_initial_diagnostics(R=R, L=L, C=C))
        prompt = deepcopy(buffer)
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

    def before_iteration(self, buffer: List[Dict[str, Any]], R, L, C) -> None:
        """Apply mode-specific control changes before constructing this iteration's prompt."""
        pass

    def handle_iteration_complete(self, buffer: List[Dict[str, Any]], R, L, C) -> str | None:
        """Return a terminal status when the current search should stop."""
        best_candidate = self.best_candidate()
        if (
            best_candidate is not None and best_candidate.metric("mse", "train") == 0.0
        ):
            return "early_stopped"
        return None

    def run_initial_diagnostics(self, R, L, C) -> Dict[str, str]:
        """Run the mandatory branch-opening diagnostics and format them for the LLM."""
        calls = [
            ToolCall(name="statistics_analysis", params={}),
            ToolCall(name="relationship_analysis", params={}),
            ToolCall(name="read_skill", params={"name": "discover-symbolic-laws"}),
        ]
        results = self.execute_action(calls)
        self.record_tool_calls(calls, results, R=R, L=L, C=C, forced=True)
        failures = [
            f"{call.name}: {result.result_str}"
            for call, result in zip(calls, results)
            if not result.ok
        ]
        if failures:
            raise RuntimeError(
                "Required initial diagnostic tool call(s) failed:\n" + "\n".join(failures)
            )
        sections = [
            f"## {call.name}\n{result.result_str}"
            for call, result in zip(calls, results)
        ]
        return {
            "role": "user",
            "content": (
                "[Required initial diagnostics]\n"
                "The framework ran these mandatory tools before your first response in this branch. "
                "Use their evidence and the skill instructions to plan the search.\n\n"
                + "\n\n".join(sections)
            ),
        }
    
    def request_llm(self, prompt: List[Dict[str, Any]], R, L, C):
        """请求 LLM 得到 Content 和 Tool Calls。"""
        response_list = []
        route = self.model_router.route(
            task_score=self._task_route_score,
            task_reasons=self._task_route_reasons,
            refinement_step=L,
        )
        self._last_model_route = route
        api = self.api
        if route.tier == "strong":
            if self._strong_api is None:
                self._strong_api = BaseAPI.create(
                    route.provider,
                    model=route.model,
                    tool_list=self.tools,
                    tool_parser_name=self.tool_parser,
                )
            api = self._strong_api
        _logger.info(
            f"Model route: tier={route.tier}, backend={route.provider}/{route.model}, "
            f"score={route.score}, reason={route.reason}"
        )
        llm_result = api(prompt, n=self.local_sample_size, max_tokens=self.llm_max_tokens)
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
        usage = self.record_llm_result(llm_result, R=R, L=L, C=C)
        return response_list, usage
    
    def get_results(self, response_list, R, L, C):
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
        self.record_tool_calls(all_tool_calls, all_results, R=R, L=L, C=C)
        return results_list

    def record_tool_calls(
        self, tool_calls: List[ToolCall],
        results: List[ToolCallResult],
        R, L, C, forced: bool = False
    ) -> None:
        """Persist tool calls from either the LLM or framework-enforced diagnostics."""
        if self.save_path is None:
            return
        with open(Path(self.save_path) / 'tool_calls.jsonl', 'a') as f:
            for tool_call, result in zip(tool_calls, results):
                json.dump({
                    "progress": self.format_progress(R, L, C),
                    "forced": forced,
                    'name': tool_call.name,
                    'params': tool_call.params,
                    "ok": result.ok,
                    "result": result.result,
                    "result_str": result.result_str,
                    "meta_data": result.meta_data,
                }, f)
                f.write('\n')
    
    def update_buffer(
        self,
        buffer: List[Dict[str, Any]],
        response_list: List[Tuple[str, List[ToolCall], Dict[str, Any]]],
        results_list: List[List[ToolCallResult]],
        node_parents: Dict[str, str], R, L, C
    ):
        """根据 LLM Response 和 Tool Results 更新 Buffer。"""
        # 如果没有成功的回复，跳过本轮更新
        if len(response_list) == 0:
            return buffer, node_parents
        node_parents = {}
        # 选择产生了最佳排序指标的 tool_call 所在的 response 分支
        selected_K = 1
        selected_priority = float('inf')
        for K, results in enumerate(results_list, 1):
            for result in results:
                if (priorities := self.sortby(result.result)) is not None and priorities[0] < selected_priority:
                    selected_K = K
                    selected_priority = priorities[0]
        node_parents[self.run_state.node_id(R=R, C=C, L=L, K=selected_K)] = 'continuation'
        _logger.info(f"Selected LLM branch: {selected_K}/{len(results_list)}")
        _, tool_calls, message = response_list[selected_K - 1]
        results = results_list[selected_K - 1]
        tool_calls = tool_calls.copy()
        message = deepcopy(message)
        results = results.copy()
        # 将其他 (tool_call, result) pairs 中不涉及 formula & metrics 的 pair 也加入 buffer, 以免丢失有用信息
        for K, ((extra_content, extra_tool_calls, extra_message), extra_results) in enumerate(zip(response_list, results_list), 1):
            for tool_call, result in zip(extra_tool_calls, extra_results):
                # 只考虑不涉及 formula & metrics 的工具调用结果
                if K == selected_K or result.get('data_split_results', {}).get('train', {}).get('metrics') is not None:
                    continue
                tool_calls.append(tool_call)
                results.append(result)
                # 对于 openai parser, 将 extra_message['tool_calls'] 拼到 message 中
                if self.tool_parser == 'openai':
                    if not isinstance(message.get('tool_calls'), list):
                        message['tool_calls'] = []
                    message['tool_calls'].append(tool_call.raw)
                # 对于 non-openai parser, 将 tool_calls 拼到 content 中
                else:
                    message['content'] += "\n\n" + tool_call.raw_str
                node_parents[self.run_state.node_id(R=R, C=C, L=L, K=K)] = 'context_merge'
        # 将 message 和 (tool_call, result) pairs 加入 buffer
        if tool_calls or (message.get('content') or '').strip():
            buffer.append(message)
            buffer.extend(self.parser.format_tool_result_messages(tool_calls, results))
        else:
            _logger.warning("Skipping empty LLM response (no content nor tool calls).")
        process_message = self.build_process_message(L)
        buffer.append(process_message)
        return buffer, node_parents

    def build_process_message(self, L):
        # 将当前搜索进度加入 buffer
        remaining_rounds = self.max_refinement_depth - L - 1
        progress_line = (
            f"Current progress: refinement round L={L+1}/{self.max_refinement_depth}. "
            f"From now on, {remaining_rounds} refinement round(s) remain in this branch."
        )
        if remaining_rounds > 1:
            policy = (
                "Plan tool use within the remaining refinement budget. "
                "Prefer targeted actions that can lead to a simpler and lower-MSE formula."
            )
        elif remaining_rounds == 1:
            policy = (
                "Only one refinement round remains after this response. Use at most a tightly targeted "
                "tool call now, and preserve a concrete formula candidate so the next round can submit it."
            )
        else:
            policy = (
                "This is the final refinement round for this branch. Do not spend this response on "
                "new data exploration, broad searches, or diagnostic-only evaluations. Submit or state "
                "your best available target formula now using the final-answer mechanism available in "
                "this environment, with a brief justification if text is required."
            )
        pareto_front = self.get_pareto_front()
        pareto_front_str = format_pareto_front(
            [self.candidate_dict(record) for record in pareto_front],
            concise=True,
            formula_max_length=160,
        )
        diagnostics = []
        for record in pareto_front:
            if eic := record.details.get("eic_diagnostics"):
                diagnostics.append(
                    f"- {record.formula}: EIC={eic['eic']:.6g}; "
                    f"worst subtree={eic['worst_subtree']}"
                )
        available_tools = {tool.metadata.name for tool in getattr(self, "tools", [])}
        routing = []
        distinct_formulas = list(dict.fromkeys(
            record.formula for record in self.run_state.ranked_candidates()
        ))
        if (
            len(distinct_formulas) >= 2
            and "delegate_subagent" in available_tools
            and self.tools_counter.named_count.get("delegate_subagent", 0) == 0
            and remaining_rounds >= 1
        ):
            routing.append(
                "There are multiple viable formulas. Use delegate_subagent in "
                "candidate_critique mode with the leading formulas and their metrics, then "
                "run the cheapest discriminating check it recommends."
            )
        diagnostics_text = (
            "\n\n[Candidate structural diagnostics]\n" + "\n".join(diagnostics)
            if diagnostics else ""
        )
        routing_text = (
            "\n\n[Specialist routing]\n" + "\n".join(f"- {item}" for item in routing)
            if routing else ""
        )
        return {"role": "user", "content": (
            f"[Iteration status]\n"
            f"{progress_line} {policy}\n\n"
            f"[Current Pareto Front]\n"
            f"{pareto_front_str}"
            f"{diagnostics_text}"
            f"{routing_text}"
        )}

    def push_candidate(self, record: CandidateRecord) -> None:
        if not self.run_state.push_candidate(record):
            _logger.warning(
                "Skipping candidate with missing or non-finite ranking metrics: "
                f"{record.formula!r}"
            )

    def collect_candidates(self, response_list, results_list, R, L, C):
        """Validate candidate tool results and add them to the run state."""
        loader = []
        for K in range(1, len(response_list) + 1):
            for _, res in zip(response_list[K - 1][1], results_list[K - 1]):
                if res.result.get('is_candidate'):
                    loader.append((K, res))

        for K, res in loader:
            train_metrics = res.result['data_split_results']['train']['metrics']
            assert self.ranking_metric in train_metrics, f"Tool result must contain '{self.ranking_metric}' in metrics for candidate formulas."
            assert 'complexity' in train_metrics, "Tool result must contain 'complexity' in metrics for candidate formulas."
            details = {
                key: value for key, value in res.result.items()
                if key not in {"formula", "is_candidate", "all_formulas"}
            }
            record = CandidateRecord(
                formula=res.result['formula'],
                node_id=self.run_state.node_id(R=R, C=C, L=L, K=K),
                details=details,
            )
            if "eic_diagnostics" not in res.result and not str(res.result.get("method", "")).startswith("ND2"):
                eic_tool = next((tool for tool in self.tools if tool.metadata.name == "evaluate_eic"), None)
                if eic_tool is not None:
                    audit_call = ToolCall(name="evaluate_eic", params={"f": record.formula, "repeats": 4})
                    audit_result = eic_tool(**audit_call.params)
                    self.tools_counter.add("evaluate_eic")
                    self.record_tool_calls([audit_call], [audit_result], R=R, L=L, C=C, forced=True)
                    if audit_result.ok:
                        record.details["eic_diagnostics"] = audit_result.result["eic_diagnostics"]
            if diagnostics := res.result.get("eic_diagnostics"):
                record.details["eic_diagnostics"] = diagnostics
                self.run_state.update_diagnostics(record.formula, diagnostics)
            self.push_candidate(record)
            # 对于 call_pysr 等工具，可能会返回多个 candidate formulas, 可以将它们全部加入 top-k
            for formula_dict in res.result.get('all_formulas', []):
                assert self.ranking_metric in formula_dict["data_split_results"]["train"]["metrics"], (
                    f"Tool result must contain '{self.ranking_metric}' in metrics for candidate formulas."
                )
                record = CandidateRecord(
                    formula=formula_dict["formula"],
                    node_id=self.run_state.node_id(R=R, C=C, L=L, K=K),
                    details={key: value for key, value in formula_dict.items() if key != "formula"},
                )
                self.push_candidate(record)
        return self.run_state.ranked_candidates()
    
    def log_info(self, response_list, R, L, C):
        """打印本轮日志, response_list 是用来统计本轮新增工具调用次数的。"""
        new_count = defaultdict(int)
        for _, tool_calls, _ in response_list:
            for tool_call in tool_calls:
                new_count[tool_call.name] += 1
        tool_calls_str = ', '.join(
            f"{name}: {count} ({new_count[name]} new)" 
            for name, count in self.tools_counter.named_count.items()
        )
        if best_record := self.best_candidate():
            metric_label, metric_value = self.record_metric(best_record)
            best_metric = f"{best_record.formula} ({metric_label}={metric_value:.6g})"
        else:
            best_metric = "None"
        log = {
            "Progress": self.format_progress(R, L, C),
            "Best": best_metric,
            "Tool Calls": tool_calls_str,
            "Speed": self.total_timer.to_str('pace'),
            "Time Usage": self.named_timer.to_str('time', 'pace', 'by_time'),
            "Token Usage": self.token_counter.to_str('count', 'speed', 'by_count'),
            "Price Usage": self.money_counter.to_str('count', 'speed', 'by_count'),
        }
        msg = "[gray] | [reset]".join(f"[blue]{k}[reset]={v}" for k, v in log.items())
        _logger.info(tag2ansi(msg))

    def record_llm_result(self, llm_result, R, L, C) -> Dict[str, Any] | None:
        """记录最近一次 LLM 请求的返回值和用量统计。"""
        usage = llm_result.returned['usage']
        for name, num in usage['token'].items():
            self.token_counter.add(name, num)
        for name, num in usage['price'].items():
            self.money_counter.add(name, num)
        if self.save_path is not None:
            model_router = vars(self._last_model_route) if self._last_model_route is not None else None
            with open(Path(self.save_path) / 'response.jsonl', 'a') as f:
                json.dump({
                    "responses": llm_result.returned["responses"],
                    "progress": self.format_progress(R, L, C),
                    "usage": usage,
                    "model_route": model_router
                }, f)
                f.write('\n')
        return usage
    
    def record_search_iteration(
        self, response_list, results_list, parent_nodes, prompt, usage, R, L, C,
    ):
        """Record one visualization node for each local sample."""
        parents: list[ParentLink] = []
        for parent_node_id, relation in parent_nodes.items():
            if relation not in {"restart_seed", "continuation", "context_merge"}:
                _logger.warning(f"Unknown parent relation: {relation} for node_id: {parent_node_id}.")
                continue
            parents.append(self.run_state.parent_link(parent_node_id, relation))

        self.run_state.register_iteration(response_list, results_list, tuple(parents), prompt, usage, R, L, C)

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

    def format_progress(self, R, L, C):
        return (
            f"(R={R}/{self.max_restart_loop}) × "
            f"(C={C}/{self.global_width}) × "
            f"(L={L}/{self.max_refinement_depth}) × "
            f"(K={self.local_sample_size})"
        )

    def record_metric(self, record):
        metric_label = self.ranking_metric.replace('_', ' ').upper()
        if isinstance(record, CandidateRecord):
            split_results = record.details.get("data_split_results")
        elif isinstance(record, dict):
            split_results = record.get("data_split_results")
        else:
            return None, None
        if not isinstance(split_results, dict):
            return None, None
        validation_metrics = split_results.get("validation", {}).get("metrics", {})
        if (metric_value := validation_metrics.get(self.ranking_metric)) is not None:
            return f"validation {metric_label}", metric_value
        train_metrics = split_results.get("train", {}).get("metrics", {})
        if (metric_value := train_metrics.get(self.ranking_metric)) is not None:
            return f"training {metric_label}", metric_value
        return None, None

    def sortby(self, record):
        _, metric_value = self.record_metric(record)
        if metric_value is None:
            return None

        split_results = (
            record.details.get('data_split_results', {})
            if isinstance(record, CandidateRecord)
            else record.get('data_split_results', {})
        )
        train_metrics = split_results.get('train', {}).get('metrics', {})
        validation_metrics = split_results.get('validation', {}).get('metrics', {})
        metric_values = [train_metrics.get(self.ranking_metric)]
        if self.ranking_metric in validation_metrics:
            metric_values.append(validation_metrics[self.ranking_metric])
        complexity = train_metrics.get('complexity', float('inf'))
        try:
            finite_values = [float(value) for value in metric_values]
            finite_complexity = float(complexity)
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(value) for value in finite_values) or not np.isfinite(finite_complexity):
            return None

        return (
            -float(metric_value) if self.larger_is_better else float(metric_value),
            finite_complexity,
        )

    @staticmethod
    def candidate_dict(record: CandidateRecord) -> dict[str, Any]:
        """Adapt a candidate to utilities that consume split results at top level."""
        return {
            "formula": record.formula,
            "node_id": record.node_id,
            **record.details,
        }

    def best_candidate(self) -> CandidateRecord | None:
        candidates = self.run_state.ranked_candidates()
        return candidates[0] if candidates else None

    def get_pareto_front(self) -> list[CandidateRecord]:
        """Return candidates on the metric-complexity Pareto front."""
        candidates = self.run_state.ranked_candidates()
        return [candidates[index] for index in self.run_state.pareto_indices(candidates)]

    def search_result(self, status: str, R: int | None, L: int | None, C: int | None):
        progress = self.format_progress(R, L, C)
        return self.run_state.result(status=status, progress=progress).to_dict()
