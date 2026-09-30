# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Effective Information Criterion diagnostics for expression subtrees."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import nd2py as nd
import numpy as np

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("evaluate_eic")
class EICTool(BaseTool):
    """Diagnose numerical information loss throughout an expression tree."""

    metadata = ToolMetadata(name="evaluate_eic")

    def execute(
        self,
        f: str,
        noise_level: float = 0.001,
        repeats: int = 8,
        random_state: int = 0,
        zero_epsilon: float = 1e-6,
    ) -> Dict[str, Any]:
        """Evaluate whole-formula and per-subtree EIC diagnostics.

        Relative Gaussian noise is injected after every non-leaf operation, following the EIC
        paper's recursive algorithm. The formula EIC is the maximum over all subtree EIC values,
        which exposes unstable internal structures even when an outer operation masks them.

        Args:
            f: Formula to assess.
            noise_level: Relative Gaussian noise injected after each operation.
            repeats: Independent perturbation runs to average, between 1 and 100.
            random_state: Random seed for reproducibility.
            zero_epsilon: Denominator fallback used only where a clean subtree output is zero.
        """
        if not 0 < noise_level < 1:
            raise ValueError("noise_level must be between 0 and 1")
        if zero_epsilon <= 0:
            raise ValueError("zero_epsilon must be positive")
        repeats = max(1, min(int(repeats), 100))
        formula = self.parse_formula(f)
        data = self.context["data"]
        paths = self._collect_paths(formula)
        samples: dict[tuple[int, ...], list[float]] = defaultdict(list)
        finite: dict[tuple[int, ...], list[float]] = defaultdict(list)
        overall_samples = []
        rng = np.random.default_rng(random_state)

        for _ in range(repeats):
            run: dict[tuple[int, ...], dict[str, Any]] = {}
            self._evaluate_recursive(
                formula, data, rng, noise_level, zero_epsilon, (), run
            )
            for path, values in run.items():
                samples[path].append(values["eic"])
                finite[path].append(values["finite_fraction"])
            overall_samples.append(max(values["eic"] for values in run.values()))

        clean_output = np.asarray(formula.eval(data), dtype=float)
        subtree_stats = []
        mean_eic = {path: float(np.mean(values)) for path, values in samples.items()}
        for path, node in paths:
            child_paths = [path + (i,) for i in range(len(node.operands))]
            child_peak = max((mean_eic[p] for p in child_paths), default=0.0)
            descendant_peak = max(
                (
                    value
                    for descendant_path, value in mean_eic.items()
                    if len(descendant_path) > len(path)
                    and descendant_path[:len(path)] == path
                ),
                default=0.0,
            )
            node_eic = mean_eic[path]
            impact = self._output_impact(formula, path, data, clean_output)
            diagnosis = self._diagnose(
                node_eic,
                child_peak,
                descendant_peak,
                impact,
                bool(node.operands),
            )
            subtree_stats.append({
                "path": "root" if not path else "root/" + "/".join(map(str, path)),
                "expression": node.to_str(number_format=self.FORMULA_DISPLAY_NUMBER_FORMAT),
                "node_type": type(node).__name__,
                "eic": node_eic,
                "eic_std": float(np.std(samples[path])),
                "introduced_eic": max(0.0, node_eic - child_peak),
                "output_impact": impact,
                "finite_fraction": float(np.mean(finite[path])),
                "diagnosis": diagnosis,
            })

        output = subtree_stats[0]
        worst = max(subtree_stats, key=lambda item: item["eic"])
        result = {
            "formula": formula.to_str(number_format=self.FORMULA_DISPLAY_NUMBER_FORMAT),
            "eic": float(np.mean(overall_samples)),
            "eic_std": float(np.std(overall_samples)),
            "output_eic": output["eic"],
            "output_eic_std": output["eic_std"],
            "noise_level": noise_level,
            "zero_epsilon": zero_epsilon,
            "repeats": repeats,
            "finite_fraction": output["finite_fraction"],
            "worst_subtree": worst["path"],
            "subtrees": subtree_stats,
            "tree": self._annotated_tree(formula, subtree_stats),
            "implementation": "recursive operator-noise Monte Carlo with subtree diagnostics",
            "source": "https://github.com/tsinghua-fib-lab/EIC",
        }
        target = self.context.get("target")
        if target and self.context.get("data"):
            result |= self.evaluate(
                f=formula,
                y=self.parse_formula(target),
                show_diagnostics=False,
            )
            result["eic_diagnostics"] = {
                key: result[key]
                for key in (
                    "eic", "eic_std", "output_eic", "output_eic_std",
                    "worst_subtree", "tree",
                )
            }
        return result

    def _evaluate_recursive(
        self, node, data, rng, noise_level, zero_epsilon, path, records
    ):
        if isinstance(node, nd.Variable):
            clear = np.asarray(data[node.name], dtype=float)
            noisy = clear
            eic = 0.0
        elif isinstance(node, nd.Number):
            clear = np.asarray(node.value, dtype=float)
            noisy = clear
            eic = 0.0
        else:
            operands = [
                self._evaluate_recursive(
                    child, data, rng, noise_level, zero_epsilon, path + (i,), records
                )
                for i, child in enumerate(node.operands)
            ]
            clear = self._apply_node(node, [item[0] for item in operands])
            noisy = self._apply_node(node, [item[1] for item in operands])
            noisy = noisy + noise_level * noisy * rng.normal(size=np.shape(noisy))
            eic = self._eic(clear, noisy, noise_level, zero_epsilon)
        records[path] = {
            "eic": eic,
            "finite_fraction": float(np.mean(np.isfinite(clear))),
        }
        return clear, noisy

    @staticmethod
    def _apply_node(node, operands):
        local = node.copy()
        local_data = {}
        for index, _ in enumerate(list(local.operands)):
            name = f"eic_operand_{index}"
            local.operands[index] = nd.Variable(name)
            local_data[name] = operands[index]
        with np.errstate(all="ignore"):
            return np.asarray(local.eval(local_data), dtype=float)

    @staticmethod
    def _eic(clear, noisy, noise_level: float, zero_epsilon: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = clear + zero_epsilon * (clear == 0)
            relative = (noisy - clear) / denominator
            valid = relative[np.isfinite(relative)]
            amplification = np.var(valid) / (noise_level**2) if valid.size else np.nan
        if not np.isfinite(amplification):
            return 30.0
        amplification = float(np.clip(amplification, 1e-30, 1e30))
        return max(0.0, 0.5 * np.log10(amplification))

    @staticmethod
    def _collect_paths(formula):
        result = []

        def visit(node, path):
            result.append((path, node))
            for index, child in enumerate(node.operands):
                visit(child, path + (index,))

        visit(formula, ())
        return result

    @staticmethod
    def _output_impact(formula, path, data, baseline) -> float | None:
        """Estimate subtree relevance by replacing one path with zero."""
        try:
            candidate = nd.parse(formula.to_str())
            if not path:
                masked = nd.Number(0)
            else:
                parent = candidate
                for index in path[:-1]:
                    parent = parent.operands[index]
                parent.operands[path[-1]] = nd.Number(0)
                masked = candidate
            prediction = np.asarray(masked.eval(data), dtype=float)
            prediction = np.broadcast_to(prediction, np.shape(baseline))
            valid = np.isfinite(baseline) & np.isfinite(prediction)
            if not np.any(valid):
                return None
            scale = float(np.sqrt(np.mean(np.square(baseline[valid]))))
            error = float(np.sqrt(np.mean(np.square(prediction[valid] - baseline[valid]))))
            return error / max(scale, 1e-300)
        except Exception:
            return None

    @staticmethod
    def _diagnose(
        eic: float,
        child_peak: float,
        descendant_peak: float,
        impact: float | None,
        is_operator: bool,
    ) -> str:
        if not is_operator:
            return "leaf input/constant; no operator noise injected"
        introduced = eic - child_peak
        if impact is not None and impact < 1e-6:
            if descendant_peak >= 1:
                return (
                    "negligible output impact masks a sensitive descendant; "
                    "likely redundant subtree"
                )
            return "negligible output impact; candidate redundant subtree"
        if descendant_peak - eic >= 1:
            return (
                "attenuates or masks a more sensitive descendant; "
                "inspect structural redundancy"
            )
        if eic >= 3:
            if introduced >= 0.5:
                return "severe information loss introduced at this operation"
            return "severe sensitivity inherited from a descendant subtree"
        if eic >= 1:
            if introduced >= 0.25:
                return "noticeable numerical sensitivity introduced at this operation"
            return "noticeable sensitivity propagated from descendants"
        return "numerically stable on the supplied data"

    @staticmethod
    def _annotated_tree(formula, subtrees: list[dict[str, Any]]) -> str:
        lines = formula.to_tree(number_format=BaseTool.FORMULA_DISPLAY_NUMBER_FORMAT).splitlines()
        if len(lines) != len(subtrees):
            return formula.to_tree(number_format=BaseTool.FORMULA_DISPLAY_NUMBER_FORMAT)
        annotated = []
        for line, item in zip(lines, subtrees):
            impact = item["output_impact"]
            impact_text = "n/a" if impact is None else f"{impact:.3g}"
            annotated.append(
                f"{line}  ← EIC={item['eic']:.3g}, impact={impact_text}; "
                f"{item['diagnosis']}"
            )
        return "\n".join(annotated)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        return (
            f"Formula EIC (maximum over all subtrees): {result['eic']:.6g} "
            f"± {result['eic_std']:.3g}\n"
            f"Root-output EIC: {result['output_eic']:.6g} "
            f"± {result['output_eic_std']:.3g}; worst subtree: {result['worst_subtree']}\n"
            f"Finite root outputs: {result['finite_fraction']:.1%}\n\n"
            f"Annotated expression tree:\n{result['tree']}"
        )

    @classmethod
    def get_doc(cls) -> dict[str, str]:
        return {
            "name": "eic-structural-stability",
            "description": (
                "Use EIC subtree diagnostics to reject numerically unstable or redundant "
                "candidate formula structures."
            ),
            "content": (
                "Use `evaluate_eic` after a candidate fits reasonably well. Compare candidates "
                "on identical data, noise level, repeats, and seed. Lower EIC is preferable, but "
                "EIC complements rather than replaces prediction error and complexity. Inspect "
                "the annotated tree: a high introduced EIC localizes the operation causing "
                "information loss; a high descendant EIC with low output impact indicates a "
                "masked, potentially redundant subtree. The reported formula EIC is the maximum "
                "over all subtrees, following the paper's recursive definition.\n\n"
                "Source: https://github.com/tsinghua-fib-lab/EIC"
            ),
        }


"""
Real invocation and selected output (captured with NumPy 2.2.6):

>>> EICTool(data={"x": np.linspace(1, 2, 16)}).execute(
...     "x + x**2", repeats=2, random_state=7
... )
{'formula': 'x + x ** 2', 'eic': 0.002206316240862223,
 'output_eic': 0.002206316240862223, 'worst_subtree': 'root', ...}
"""
