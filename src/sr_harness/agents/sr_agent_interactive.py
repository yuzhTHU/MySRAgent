# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""人机协同符号回归 Agent。

继承自 SRAgent，以 L（对话轮次）为搜索主体，支持人类实时干预和工作区文件操作。
"""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional

from ..api import BaseAPI
from ..interaction import InteractionManager, TerminalInteractionManager
from ..tools import BaseTool
from .sr_agent import SRAgent
from ..parser import BaseParser

_logger = logging.getLogger(f'sr_harness.{__name__}')


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
        run_id: str | None = None,
        local_sample_size: int = 1,
        max_refinement_depth: int = 50,
        global_width: int = 1,
        max_restart_loop: int = 1,
        restart_top_k: int = 1,
        max_workers: int = 0,
        validation_fraction: float = 0.2,
        split_by: str = "ood",
        split_random_state: int = 42,
        ranking_metric: str = "mse",
        larger_is_better: bool = False,
        use_workspace: bool = False,
        workspace_files: List[str | Path] | None = None,
        human_input_callback: Optional[Callable[[str], str]] = None,
        interaction_manager: InteractionManager | None = None,
        force_initial_diagnostics: bool = False,
        auto_routing: bool = True,
        strong_llm_provider: str | None = None,
        strong_llm_model: str | None = None,
    ):
        """初始化 SRAgentInteractive。

        Args:
            llm_provider: LLM 提供商名称。
            llm_model: 模型名称。
            tools: 可用工具名列表。None 表示使用默认工具集（全部工具减去 code_executor）。
            verbose: 是否启用详细日志。
            tool_parser: 工具解析器类型。
            save_path: 日志保存路径。
            run_id: 本次运行的全局唯一标识。None 表示自动生成。
            local_sample_size: 每轮 LLM 采样数量（K）。
            max_refinement_depth: 最大对话轮次（L），也是搜索深度上限。
            global_width: 独立分支数量（C）。
            max_restart_loop: 重启次数（R）。
            restart_top_k: 重启时注入历史最佳结果数量。
            max_workers: 并行工作进程数（0 表示不并行）。
            validation_fraction: 验证集比例。
            split_by: 验证集划分方式，可选 "random" 或 "ood"。
            split_random_state: 数据划分随机种子。
            ranking_metric: 候选公式排序所用的指标键；默认使用 mse。
            larger_is_better: 排序指标是否越大越好；默认按越小越好排序。
            use_workspace: 是否使用工作区。
            workspace_files: 初始化到工作区的文件/目录路径列表。
            human_input_callback: 人类输入回调函数。默认由交互管理器提供。
            interaction_manager: 连接 Agent 与 Web、终端等交互界面的管理器。
            force_initial_diagnostics: 是否在每个分支开始时强制执行初始诊断。
            auto_routing: 是否根据任务复杂度在基础与强模型后端之间自动路由。
            strong_llm_provider: 复杂任务使用的后端；默认沿用 llm_provider。
            strong_llm_model: 复杂任务使用的模型。None 表示仅使用基础模型。
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
            run_id=run_id,
            local_sample_size=local_sample_size,
            max_refinement_depth=max_refinement_depth,
            global_width=global_width,
            max_restart_loop=max_restart_loop,
            restart_top_k=restart_top_k,
            max_workers=max_workers,
            validation_fraction=validation_fraction,
            split_by=split_by,
            split_random_state=split_random_state,
            ranking_metric=ranking_metric,
            larger_is_better=larger_is_better,
            force_initial_diagnostics=force_initial_diagnostics,
            auto_routing=auto_routing,
            strong_llm_provider=strong_llm_provider,
            strong_llm_model=strong_llm_model,
        )

        # 工作区
        self.use_workspace = use_workspace
        self.workspace_files = workspace_files

        # 交互界面
        self.interaction_manager = interaction_manager or TerminalInteractionManager()
        self.interaction_manager.bind_run_state(self.run_state)
        self.human_input_callback = human_input_callback or self.interaction_manager.ask_human

    @contextmanager
    def prepare_tool_context(self, tool_context: dict[str, Any]):
        """Add interaction resources to the tool context for the duration of a run."""
        tool_context = dict(tool_context)
        tool_context["human_input_callback"] = self.human_input_callback
        if not self.use_workspace:
            yield tool_context
            return

        from ..tools.workspace_shell import Workspace

        with Workspace(self.workspace_files, self.save_path) as workspace:
            _logger.note(f"Workspace initialized at: {workspace.path}")
            tool_context.update(
                workspace=workspace,
                workspace_dir=str(workspace.path),
            )
            yield tool_context

    def before_iteration(self, buffer, R: int, L: int, C: int) -> None:
        """Apply queued human guidance before the prompt is constructed."""
        if R == C == L == 1:
            self._perfect_candidate_announced = False
        self.emit("activity", {"phase": "checkpoint", "coord": {"R": R, "C": C, "L": L}})
        for message in self.interaction_manager.checkpoint():
            buffer.append({
                "role": "user",
                "content": f"[Human guidance injected during the run]\n{message}",
            })
        if settings := self.interaction_manager.take_model_settings():
            self._apply_model_settings(settings)

    def _apply_model_settings(self, settings: dict[str, Any]) -> None:
        """Apply a frontend-requested model change at a safe iteration boundary."""
        try:
            self.api = BaseAPI.create(
                settings["llm_provider"],
                model=settings["llm_model"],
                tool_list=self.tools,
                tool_parser_name=self.tool_parser,
            )
            self.llm_provider = settings["llm_provider"]
            self.llm_model = settings["llm_model"]
            self.model_router.base_provider = self.llm_provider
            self.model_router.base_model = self.llm_model
            self.model_router.enabled = False
            self._strong_api = None
            for tool in self.tools:
                tool.context.update(settings)
            self.interaction_manager.commit_model_settings(settings)
            self.emit("settings_applied", settings)
        except Exception as exc:
            self.emit("settings_error", {"error": str(exc)})

    def handle_iteration_complete(self, buffer, R: int, L: int, C: int) -> str | None:
        """Keep interactive runs open after finding an exact candidate."""
        best_candidate = self.best_candidate()
        if (
            best_candidate is not None
            and best_candidate.metric("mse", "train") == 0.0
            and not self._perfect_candidate_announced
        ):
            self._perfect_candidate_announced = True
            buffer.append({
                "role": "user",
                "content": (
                    "Congratulations! You've found a formula with MSE=0. "
                    "Please conclude the search and call ask_human with a summary "
                    "of your discovery and the final formula."
                ),
            })
        return None

    def build_initial_prompt(self, problem_description, X, y, restart_records):
        """构建面向交互式探索的 initial prompt。

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
            for record in restart_records:
                formula = record.formula
                mse = record.metric("mse", "validation")
                if mse is None:
                    mse = record.metric("mse", "train")
                r2 = record.metric("r2", "validation")
                if r2 is None:
                    r2 = record.metric("r2", "train")
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
        for tool in self.tools:
            if (workspace := tool.context.get("workspace")) is not None:
                self.interaction_manager.bind_workspace(workspace)
                break
        return self.interaction_manager.prepare_initial_prompt(
            initial_prompt,
            X=X,
            y=y,
        )

    def request_llm(self, prompt, R: int, L: int, C: int):
        """Request the model while publishing frontend-neutral progress events."""
        coord = {"R": R, "C": C, "L": L}
        self.emit("context", {"messages": prompt, "coord": coord})
        route = self.model_router.route(
            task_score=self._task_route_score,
            task_reasons=self._task_route_reasons,
            refinement_step=L,
        )
        self.emit("activity", {
            "phase": "model",
            "coord": coord,
            "provider": route.provider,
            "model": route.model,
        })
        responses, usage = super().request_llm(prompt, R=R, L=L, C=C)
        self.emit("activity", {"phase": "processing", "coord": coord})
        cumulative_usage = {
            "token": self.token_counter.named_count,
            "price": self.money_counter.named_count,
        }
        tool_schemas = {
            tool.metadata.name: self.tool_schema(tool.metadata.name)
            for tool in self.tools
            if getattr(tool, "metadata", None) is not None
        }
        for K, (content, calls, message) in enumerate(responses, 1):
            self.emit("assistant", {
                "content": content,
                "message": message,
                "tool_calls": calls,
                "tool_schemas": tool_schemas,
                "coord": coord | {"K": K},
                "usage": usage,
                "cumulative_usage": cumulative_usage,
            })
        return responses, usage

    def tool_schema(self, name: str) -> dict[str, Any]:
        """Return the schema exposed by one initialized tool."""
        return next(
            (
                {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
                if tool.metadata.name == name
            ),
            {},
        )

    def execute_action(self, actions):
        """Execute tools serially with safe control boundaries and UI events."""
        results = []
        for action in actions:
            tool_schema = self.tool_schema(action.name)
            self.emit("activity", {"phase": "checkpoint", "tool": action.name})
            self.interaction_manager.wait_until_running()
            self.emit("activity", {"phase": "tool", "tool": action.name})
            self.emit("tool_start", {"call": action, "tool_schema": tool_schema})
            started_at = time.monotonic()
            try:
                result = super().execute_action([action])[0]
            except BaseException as exc:
                self.emit("tool_error", {
                    "call": action,
                    "tool_schema": tool_schema,
                    "duration_seconds": time.monotonic() - started_at,
                    "error": str(exc),
                })
                raise
            self.emit("tool_result", {
                "call": action,
                "tool_schema": tool_schema,
                "duration_seconds": time.monotonic() - started_at,
                "result": result,
            })
            results.append(result)
        self.emit("activity", {"phase": "processing"})
        return results

    def collect_candidates(self, *args, **kwargs):
        """Update scientific state and publish its current ranked view."""
        self.emit("activity", {"phase": "ranking"})
        records = super().collect_candidates(*args, **kwargs)
        self.emit("topk", {"records": [record.display_dict() for record in records]})
        return records

    def record_tool_calls(self, tool_calls, results, R, L, C, forced=False):
        """Persist tool calls and expose framework-enforced calls to the UI."""
        super().record_tool_calls(tool_calls, results, R=R, L=L, C=C, forced=forced)
        if forced:
            for call, result in zip(tool_calls, results):
                self.emit("tool_result", {
                    "call": call,
                    "tool_schema": self.tool_schema(call.name),
                    "result": result,
                    "forced": True,
                })

    def emit(self, kind: str, payload: Any) -> None:
        """Publish an event through the configured interaction manager."""
        self.interaction_manager.publish(kind, payload)

    def execute_action_parallel(self, actions, max_workers: int):
        raise NotImplementedError(
            "Parallel execution is not supported in interactive mode, "
            "since tools like ask_human and workspace_shell cannot guarantee read-only access. "
            "Please set max_workers=0 to disable parallel execution when using interactive tools."
        )
