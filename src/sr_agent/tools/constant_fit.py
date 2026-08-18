# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""比例常数和不变量常数拟合工具。"""
import numpy as np
from fractions import Fraction
from typing import Any, Dict
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("constant_fit")
class ConstantFitTool(BaseTool):
    metadata = ToolMetadata("constant_fit")

    def execute(
        self,
        eq: str,
        y: str = None,
        use_eq_as_y: bool = False,
        max_denominator: int = 64,
        recognition_tolerance: float = 1e-8,
        n_stability_subsets: int = 5,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Fit and recognize a scalar constant for a candidate expression.

        By default fit y = c * eq. With use_eq_as_y=true, fit eq = c to test
        whether an expression is approximately invariant.

        Args:
            eq: Numeric expression used as the multiplicative basis, or as the quantity expected to be constant.
            y: Target variable or expression. Use the formula-discovery target by default.
            use_eq_as_y: Whether to fit eq = c instead of y = c * eq.
            max_denominator: Largest denominator used for rational and named-constant recognition (1-1000).
            recognition_tolerance: Maximum relative recognition error and normalized RMSE degradation for acceptance.
            n_stability_subsets: Number of deterministic 80% subsets used to estimate constant stability (1-20).
            show_diagnostics: Whether final metrics should include compact residual diagnostics.
        """
        data = self.context["data"]
        eq = self.normalize_formula(eq)
        eq_symbol = self.parse_formula(eq)
        eq_values = np.asarray(eq_symbol.eval(data), dtype=float).flatten()

        if use_eq_as_y:
            y_symbol = eq_symbol
            observed = eq_values
            basis = np.ones_like(observed)
        else:
            target_name = (y or self.context["target"]).strip().strip('"').strip("'")
            y_symbol = self.parse_formula(target_name)
            observed = np.asarray(y_symbol.eval(data), dtype=float).flatten()
            basis = eq_values
            if observed.shape != basis.shape:
                raise ValueError("eq and y have different shapes")

        valid = np.isfinite(observed) & np.isfinite(basis)
        denominator = float(np.dot(basis[valid], basis[valid]))
        if not np.any(valid) or denominator == 0:
            raise ValueError("No finite, non-degenerate samples are available for constant fitting.")

        constant = float(np.dot(basis[valid], observed[valid]) / denominator)
        prediction = constant * basis[valid]
        formula = f"({constant:.12g})" if use_eq_as_y else f"({constant:.12g}) * ({eq})"
        evaluation = self.evaluate(
            f=self.parse_formula(formula), y=y_symbol,
            show_diagnostics=show_diagnostics,
        )

        # Search a compact set of interpretable representations of the fitted value.
        max_denominator = max(1, min(int(max_denominator), 1000))
        raw_candidates = []
        rational = Fraction(constant).limit_denominator(max_denominator)
        label = str(rational.numerator) if rational.denominator == 1 else f"{rational.numerator}/{rational.denominator}"
        raw_candidates.append((label, float(rational)))
        named_constants = {"pi": np.pi, "e": np.e}
        named_constants.update({f"sqrt({n})": np.sqrt(n) for n in range(2, 11)})
        for name, value in named_constants.items():
            ratio = Fraction(constant / value).limit_denominator(max_denominator)
            prefix = str(ratio.numerator) if ratio.denominator == 1 else f"({ratio.numerator}/{ratio.denominator})"
            raw_candidates.append((f"{prefix}*{name}", float(ratio) * value))
            if constant != 0:
                ratio = Fraction(constant * value).limit_denominator(max_denominator)
                prefix = str(ratio.numerator) if ratio.denominator == 1 else f"({ratio.numerator}/{ratio.denominator})"
                raw_candidates.append((f"{prefix}/{name}", float(ratio) / value))

        scale = max(abs(constant), np.finfo(float).eps)
        target_scale = max(float(np.mean(np.abs(observed[valid]))), np.finfo(float).eps)
        fitted_rmse = evaluation["data_split_results"]["train"]["metrics"]["rmse"]
        recognized = []
        for expression, value in raw_candidates:
            relative_error = abs(value - constant) / scale
            candidate_rmse = float(np.sqrt(np.mean((observed[valid] - value * basis[valid]) ** 2)))
            degradation = (candidate_rmse - fitted_rmse) / target_scale
            recognized.append({
                "expression": expression,
                "value": value,
                "relative_error": relative_error,
                "normalized_rmse_degradation": degradation,
                "accepted": bool(
                    relative_error <= max(float(recognition_tolerance), 0.0)
                    and degradation <= max(float(recognition_tolerance), 0.0)
                ),
            })
        recognized.sort(key=lambda item: item["relative_error"])

        n_stability_subsets = min(max(int(n_stability_subsets), 1), 20)
        subset_constants = []
        for repeat in range(n_stability_subsets):
            indices = np.random.default_rng(repeat).choice(
                np.count_nonzero(valid), max(1, int(0.8 * np.count_nonzero(valid))), replace=False
            )
            subset_basis = basis[valid][indices]
            subset_observed = observed[valid][indices]
            subset_denominator = float(np.dot(subset_basis, subset_basis))
            if subset_denominator:
                subset_constants.append(float(np.dot(subset_basis, subset_observed) / subset_denominator))

        accepted = [item for item in recognized if item["accepted"]]
        return evaluation | {
            "mode": "eq_as_constant" if use_eq_as_y else "scale_eq_to_y",
            "constant": constant,
            "plausible_simple_constant": accepted[0] if accepted else None,
            "subsample_constant_relative_std": (
                float(np.std(subset_constants)) / scale if subset_constants else float("nan")
            ),
            "usable_sample_fraction": float(np.mean(valid)),
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        lines = [
            (
                "Fitted relationship: expression is approximately constant."
                if result["mode"] == "eq_as_constant"
                else "Fitted relationship: target ≈ constant × input expression."
            ),
            f"Best fitted formula: {result['formula']}",
            f"Fitted constant: {result['constant']:.12g}",
        ]
        if suggestion := result["plausible_simple_constant"]:
            lines.append(
                f"Numerically plausible simplified constant: {suggestion['expression']} "
                f"(relative difference from fitted constant {suggestion['relative_error']:.3g}; "
                "accepted by the requested numerical tolerance, not proven exact)."
            )
        else:
            lines.append("Numerically plausible simplified constant: none met the requested tolerance.")
        lines.extend([
            f"Scale robustness: "
            f"{'stable' if result['subsample_constant_relative_std'] <= 0.05 else 'unstable'} across "
            f"repeated fits on resampled data subsets (relative standard deviation="
            f"{result['subsample_constant_relative_std']:.3%} of the absolute fitted coefficient; "
            "this is not a confidence interval).",
            f"Samples usable for this fit: {result['usable_sample_fraction']:.1%}.",
            cls.format_evaluation_result(result, title="Fit quality for constant relationship"),
            "This checks the supplied expression up to a constant scale; it does not test omitted "
            "variables or alternative formula structures.",
        ])
        return "\n".join(lines)
