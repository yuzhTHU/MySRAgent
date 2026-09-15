# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Explore simple replacements for numeric constants in a supplied expression."""

from fractions import Fraction
from itertools import product
from math import prod
from typing import Any, Dict

import nd2py as nd
import numpy as np

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("constant_fit")
class ConstantFitTool(BaseTool):
    metadata = ToolMetadata("constant_fit")
    NEAR_RELATIVE_TOLERANCE = 0.05
    MAX_SIMPLE_CANDIDATES_PER_NUMBER = 4
    MAX_COMBINATIONS = 50000

    def execute(self, eq: str, y: str = None, use_eq_as_y: bool = False) -> Dict[str, Any]:
        """Compare nearby simple constants at each numeric position in eq.

        Search nearby integers and fractions (denominator <= 12, numerator <= 32),
        pi, e, sqrt(2)..sqrt(10),
        and signed half/double multiples of these named constants. Candidates
        must be within 5% relative distance of the original number. The Pareto
        front maximizes validation R2 if validation data exist, otherwise train
        R2, and minimizes the count of original numeric constants left unchanged.

        Args:
            eq: Required formula containing at least one numeric constant.
            y: Target variable or expression; overrides use_eq_as_y when supplied.
            use_eq_as_y: If y is omitted, compare against original eq rather
                than the tool context's target variable.
        """
        if y is not None and not isinstance(y, str):
            raise TypeError("y must be a string expression or None.")
        original = nd.parse(
            self.normalize_formula(eq),
            variables={"pi": nd.Variable("pi"), "e": nd.Variable("e")},
        )
        leaves = [node for node in original.iter_preorder() if isinstance(node, nd.Number)]
        if not leaves:
            return {"exceptions": ["eq contains no numerical constants to replace."]}
        numbers = []
        for index, node in enumerate(leaves, 1):
            value = float(node.value)
            if not np.isfinite(value):
                return {"exceptions": [f"Number{index} is not finite."]}
            numbers.append({
                "id": f"Number{index}",
                "original": node.to_str(),
                "choices": self._nearby_constants(value, node.to_str()),
            })
        total = prod(len(item["choices"]) for item in numbers)
        if total > self.MAX_COMBINATIONS:
            return {"exceptions": [
                f"The {len(numbers)} numeric constants yield {total} combinations; "
                f"the limit is {self.MAX_COMBINATIONS}. Supply a smaller expression."
            ], "numbers": numbers}

        target_expression = y if y is not None else (
            original.to_str() if use_eq_as_y else self.context["target"]
        )
        target_symbol = self.parse_formula(target_expression)
        train = self.context["data"]
        validation = self.context.get("evaluation_data")
        train_target = self._values(target_symbol, train)
        validation_target = self._values(target_symbol, validation) if validation else None

        evaluated = []
        invalid = 0
        for choices in product(*(item["choices"] for item in numbers)):
            expression = original.copy()
            original_leaves = [node for node in expression.iter_preorder() if isinstance(node, nd.Number)]
            for leaf, choice in zip(original_leaves, choices):
                replacement = nd.parse(
                    choice["expression"],
                    variables={"pi": nd.Variable("pi"), "e": nd.Variable("e")},
                )
                expression = expression.replace(leaf, replacement, no_warn=True)
            formula = expression.to_str()
            try:
                candidate = self.parse_formula(formula)
                train_r2 = self._r2(train_target, self._values(candidate, train))
                validation_r2 = (
                    self._r2(validation_target, self._values(candidate, validation))
                    if validation else None
                )
            except (ArithmeticError, TypeError, ValueError, KeyError):
                invalid += 1
                continue
            if not np.isfinite(train_r2) or (validation and not np.isfinite(validation_r2)):
                invalid += 1
                continue
            evaluated.append({
                "formula": formula,
                "constant_complexity": sum(choice["is_original"] for choice in choices),
                "train_r2": train_r2,
                "validation_r2": validation_r2,
                "choices": {item["id"]: choice["label"] for item, choice in zip(numbers, choices)},
                "replacements": {item["id"]: choice["label"] for item, choice in zip(numbers, choices)
                                 if not choice["is_original"]},
            })

        primary = "validation_r2" if validation else "train_r2"
        best_at_complexity = {}
        for item in evaluated:
            complexity = item["constant_complexity"]
            current = best_at_complexity.get(complexity)
            if current is None or (item[primary], item["formula"]) > (current[primary], current["formula"]):
                best_at_complexity[complexity] = item
        pareto = []
        best_r2 = -float("inf")
        for complexity in sorted(best_at_complexity):
            item = best_at_complexity[complexity]
            if item[primary] > best_r2:
                pareto.append(item)
                best_r2 = item[primary]
        return {
            "target_expression": target_expression,
            "primary_metric": primary,
            "numbers": numbers,
            "pareto_front": pareto,
            "evaluated_combinations": len(evaluated),
            "invalid_combinations": invalid,
            "total_combinations": total,
            "near_relative_tolerance": self.NEAR_RELATIVE_TOLERANCE,
        }

    @classmethod
    def _nearby_constants(cls, value: float, original: str) -> list[Dict[str, Any]]:
        options = []

        def add(label: str, expression: str, candidate: float):
            distance = abs(candidate - value) / max(abs(value), np.finfo(float).tiny)
            if distance <= cls.NEAR_RELATIVE_TOLERANCE and expression not in {x["expression"] for x in options}:
                options.append({"label": label, "expression": expression, "value": candidate,
                                "relative_distance": distance, "is_original": False})

        integer = round(value)
        if abs(integer) <= 1000:
            add(str(integer), str(integer), float(integer))
        fraction = Fraction(value).limit_denominator(12)
        if fraction.denominator != 1 and abs(fraction.numerator) <= 32:
            rational = f"Number({fraction.numerator}) / Number({fraction.denominator})"
            add(str(fraction), rational, float(fraction))
        named = {"pi": np.pi, "e": np.e}
        named.update({f"sqrt({n})": np.sqrt(n) for n in range(2, 11) if n not in (4, 9)})
        for label, base in named.items():
            for factor, prefix in ((0.5, "0.5*"), (1.0, ""), (2.0, "2*"),
                                   (-0.5, "-0.5*"), (-1.0, "-"), (-2.0, "-2*")):
                add(f"{prefix}{label}", f"{prefix}{label}", factor * base)
        options.sort(key=lambda x: (x["relative_distance"], len(x["expression"]), x["expression"]))
        return [{"label": original, "expression": original, "value": value, "relative_distance": 0.0,
                 "is_original": True}, *options[:cls.MAX_SIMPLE_CANDIDATES_PER_NUMBER]]

    @staticmethod
    def _values(symbol: nd.Symbol, data: Dict[str, Any]) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(symbol.eval(data), dtype=float).ravel()

    @staticmethod
    def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
        if prediction.size == 1 and target.size != 1:
            prediction = np.full_like(target, float(prediction[0]))
        if prediction.shape != target.shape or target.size == 0:
            return float("nan")
        if not np.all(np.isfinite(target) & np.isfinite(prediction)):
            return float("nan")
        denominator = float(np.sum((target - np.mean(target)) ** 2))
        numerator = float(np.sum((target - prediction) ** 2))
        if denominator == 0:
            return 1.0 if numerator == 0 else float("nan")
        return float(1 - numerator / denominator)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if "exceptions" in result:
            return "Constant fitting not applicable: " + "; ".join(result["exceptions"])

        def format_r2(value: float) -> str:
            displayed = format(value, "#.3g").removesuffix(".")
            if value < 1 and float(displayed) == 1:
                for precision in range(4, 18):
                    displayed = format(value, f"#.{precision}g").removesuffix(".")
                    if float(displayed) < 1:
                        break
            return displayed

        lines = [
            f"Target expression: {result['target_expression']}",
            f"Nearby constants (relative distance <= {result['near_relative_tolerance']:.1%}; "
            f"at most {cls.MAX_SIMPLE_CANDIDATES_PER_NUMBER} simple candidates per Number):",
        ]
        for item in result["numbers"]:
            labels = [choice["label"] for choice in item["choices"]
                      if not choice["is_original"] and choice["label"] != item["original"]]
            if labels:
                lines.append(f"    {item['original']} -> {labels}")
        lines.append(
            f"Evaluated {result['evaluated_combinations']}/{result['total_combinations']} combinations; "
            f"invalid={result['invalid_combinations']}."
        )
        metric_name = "Validation-set R2" if result["primary_metric"] == "validation_r2" else "Train-set R2"
        lines.append(f"Pareto front (maximize #Simplified Constants and {metric_name}):")
        lines.append("    (#Simplified Constants | Train-set R2 | Validation-set R2 | Formula)")
        for item in result["pareto_front"]:
            validation = format_r2(item["validation_r2"]) if item["validation_r2"] is not None else "N/A"
            simplified = len(result["numbers"]) - item["constant_complexity"]
            lines.append(
                f"    {simplified} | {format_r2(item['train_r2'])} | {validation} | {item['formula']}"
            )
        return "\n".join(lines)
