# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Context-isolated symbolic-regression hypothesis and critique agent."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("delegate_subagent")
class SubagentTool(BaseTool):
    metadata = ToolMetadata(name="delegate_subagent")
    MODES = {
        "hypothesis_generation": (
            "Generate 2-4 structurally distinct, testable formula families. For each, state "
            "the invariant/transformation that would support it and the cheapest main-agent "
            "tool call that can falsify it. Avoid guessing constants without evidence."
        ),
        "candidate_critique": (
            "Audit the supplied candidate formulas independently. Check identifiability, domain, "
            "units when available, extrapolation, numerical stability, redundant subtrees, and "
            "whether simpler algebraically equivalent forms exist. Rank concrete follow-up tests."
        ),
        "residual_diagnosis": (
            "Use the supplied residual evidence to identify missing variables, factors, poles, "
            "symmetries, transforms, or numerical artifacts. Propose a bounded next experiment."
        ),
        "search_recovery": (
            "Assume the main search is stagnating. Challenge its current assumptions, identify "
            "an unexplored formula family, and give a short recovery branch with stop criteria."
        ),
    }

    def execute(
        self,
        objective: str,
        mode: str = "hypothesis_generation",
        candidate_formulas: List[str] = None,
        evidence: str = "",
    ) -> Dict[str, Any]:
        """Run a bounded independent SR analysis with a specific scientific role.

        Use this only when diversity or an independent audit is valuable: generating competing
        hypotheses before committing search budget, critiquing near-tied candidates, diagnosing
        structured residuals, or recovering a stagnated search. The subagent cannot execute tools
        or mutate the main search state; it must return falsifiable recommendations.

        Args:
            objective: Precise scientific question or search decision to resolve.
            mode: hypothesis_generation, candidate_critique, residual_diagnosis, or search_recovery.
            candidate_formulas: Candidate expressions to compare when relevant.
            evidence: Compact metrics, residual summaries, units, or search history from main tools.
        """
        if not objective.strip():
            raise ValueError("objective must not be empty")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of: {', '.join(self.MODES)}")
        messages = self._messages(objective, mode, candidate_formulas or [], evidence)
        callback = self.context.get("subagent_callback")
        if callback is not None:
            response = callback(messages)
            return {
                "mode": mode,
                "objective": objective,
                "content": str(response),
                "usage": {},
            }

        from ..api import LLMAPI

        api = LLMAPI.create(
            self.context.get("subagent_llm_provider") or self.context.get("llm_provider"),
            self.context.get("subagent_llm_model") or self.context.get("llm_model"),
            tool_list=None,
        )
        result = api(messages, n=1, max_tokens=self.context.get("llm_max_tokens", 4096))
        content = ""
        for content, _, _ in result:
            pass
        return {
            "mode": mode,
            "objective": objective,
            "content": content,
            "usage": result.usage,
        }

    def _messages(
        self,
        objective: str,
        mode: str,
        candidate_formulas: List[str],
        evidence: str,
    ) -> list[dict[str, str]]:
        tool_catalog = self.context.get("tool_catalog") or BaseTool.load_tool_classes()
        tool_names = ", ".join(
            sorted(
                tool.metadata.name
                for tool in tool_catalog
                if tool.metadata.name != self.metadata.name
            )
        )
        candidates = "\n".join(f"- {formula}" for formula in candidate_formulas) or "(none)"
        return [
            {
                "role": "system",
                "content": (
                    "You are an independent symbolic-regression research critic, not a generic "
                    "chat assistant. Do not repeat the main agent's conclusion by default. Return "
                    "falsifiable hypotheses or objections and a bounded experiment plan. Never "
                    "claim to have run a tool or inspected raw rows not included in the prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Mode: {mode}\nMode mandate: {self.MODES[mode]}\n\n"
                    f"Objective:\n{objective}\n\n"
                    f"Automatic data summary:\n{self._data_summary()}\n\n"
                    f"Candidate formulas:\n{candidates}\n\n"
                    f"Main-agent evidence:\n{evidence.strip() or '(none supplied)'}\n\n"
                    f"Available main-agent tools (recommend only these):\n{tool_names}"
                ),
            },
        ]

    def _data_summary(self) -> str:
        data = self.context.get("data") or {}
        target = self.context.get("target")
        target_values = self._finite_vector(data.get(target)) if target in data else None
        lines = []
        for name, values in data.items():
            vector = self._finite_vector(values)
            if vector is None or vector.size == 0:
                lines.append(f"- {name}: shape={np.shape(values)}, no finite scalar summary")
                continue
            correlation = ""
            if target_values is not None and name != target and len(vector) == len(target_values):
                if np.std(vector) > 0 and np.std(target_values) > 0:
                    value = np.corrcoef(vector, target_values)[0, 1]
                    correlation = f", pearson_to_target={value:.4g}"
            lines.append(
                f"- {name}: n={vector.size}, min={np.min(vector):.4g}, "
                f"median={np.median(vector):.4g}, max={np.max(vector):.4g}, "
                f"mean={np.mean(vector):.4g}, std={np.std(vector):.4g}{correlation}"
            )
        return "\n".join(lines) or "(data not provided)"

    @staticmethod
    def _finite_vector(values):
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim != 1:
            return None
        return array[np.isfinite(array)].astype(float, copy=False)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        return f"Independent SR subagent ({result['mode']}):\n{result['content']}"


"""
Real callback-backed invocation and output:

>>> tool = SubagentTool(
...     data={"x": np.arange(3.0), "y": np.arange(3.0)},
...     target="y",
...     subagent_callback=lambda messages: "Test x first; reject if residuals curve.",
... )
>>> tool.execute("Audit y=x", mode="candidate_critique", candidate_formulas=["x"])
{'mode': 'candidate_critique', 'objective': 'Audit y=x',
 'content': 'Test x first; reject if residuals curve.', 'usage': {}}
"""
