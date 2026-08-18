# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""乘法幂律拟合工具。"""
import numpy as np
from fractions import Fraction
from typing import Any, Dict, List
from .base_tool import BaseTool, ToolMetadata, is_numeric_array


@BaseTool.register("power_law_fit")
class PowerLawFitTool(BaseTool):
    metadata = ToolMetadata("power_law_fit")

    def execute(
        self,
        x: List[str] = None,
        y: str = None,
        include_scale: bool = True,
        snap_exponents: bool = False,
        max_denominator: int = 8,
        snap_tolerance: float = 0.05,
        max_snap_rmse_degradation: float = 0.01,
        n_stability_subsets: int = 5,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Fit a multiplicative power law y = c * product(x_i ** p_i) in log space.

        Args:
            x: Positive input variables or expressions. Use all numeric non-target variables by default.
            y: Nonzero target variable or expression. Use the formula-discovery target by default.
            include_scale: Whether to fit the multiplicative scale c.
            snap_exponents: Whether to try nearby simple rational exponents and refit c.
            max_denominator: Largest denominator considered when snapping exponents (1-32).
            snap_tolerance: Maximum distance from every fitted exponent to its snapped value.
            max_snap_rmse_degradation: Maximum normalized RMSE degradation allowed after snapping.
            n_stability_subsets: Number of deterministic 80% subsets used to estimate exponent stability (1-20).
            show_diagnostics: Whether final metrics should include compact residual diagnostics.
        """
        data = self.context["data"]
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        x = x or [name for name, value in data.items() if name != target_name and is_numeric_array(value)]
        y_symbol = self.parse_formula(target_name)
        target = np.asarray(y_symbol.eval(data), dtype=float).flatten()

        features = []
        for expression in x:
            value = data[expression] if expression in data else self.parse_formula(expression).eval(data)
            if not is_numeric_array(value):
                raise ValueError(f"'{expression}' did not produce numeric values")
            value = np.asarray(value, dtype=float).flatten()
            if value.shape != target.shape:
                raise ValueError(f"Feature '{expression}' shape does not match target shape.")
            features.append((expression, value))
        if not features:
            raise ValueError("No valid input variables available for fitting.")

        signs = np.sign(target[np.isfinite(target) & (target != 0)])
        if len(signs) == 0 or not np.all(signs == signs[0]):
            raise ValueError("Power-law fitting requires a nonzero target with a constant sign.")
        valid = np.isfinite(target) & (target != 0)
        for _, values in features:
            valid &= np.isfinite(values) & (values > 0)
        if np.count_nonzero(valid) < len(features) + int(include_scale):
            raise ValueError("Too few jointly valid samples; every fitted x must be positive.")

        log_x = np.column_stack([np.log(values[valid]) for _, values in features])
        log_y = np.log(np.abs(target[valid]))
        design = np.column_stack([np.ones(len(log_x)), log_x]) if include_scale else log_x
        coefficients, _, rank, _ = np.linalg.lstsq(design, log_y, rcond=None)
        if include_scale:
            log_scale, raw_exponents = coefficients[0], coefficients[1:]
        else:
            log_scale, raw_exponents = 0.0, coefficients
        raw_scale = float(signs[0]) * float(np.exp(log_scale))

        max_denominator = max(1, min(int(max_denominator), 32))
        snapped_exponents = np.asarray([
            float(Fraction(float(value)).limit_denominator(max_denominator))
            for value in raw_exponents
        ])
        snapped_log_scale = (
            float(np.mean(log_y - log_x @ snapped_exponents)) if include_scale else 0.0
        )
        snapped_scale = float(signs[0]) * float(np.exp(snapped_log_scale))

        raw_prediction = self._predict(features, valid, raw_exponents, raw_scale)
        snapped_prediction = self._predict(features, valid, snapped_exponents, snapped_scale)
        raw_rmse = float(np.sqrt(np.mean((raw_prediction - target[valid]) ** 2)))
        snapped_rmse = float(np.sqrt(np.mean((snapped_prediction - target[valid]) ** 2)))
        target_rms = max(float(np.sqrt(np.mean(target[valid] ** 2))), np.finfo(float).eps)
        degradation = (snapped_rmse - raw_rmse) / target_rms
        distances = np.abs(raw_exponents - snapped_exponents)
        snap_accepted = bool(
            snap_exponents
            and np.all(distances <= max(float(snap_tolerance), 0.0))
            and degradation <= max(float(max_snap_rmse_degradation), 0.0)
        )

        exponents = snapped_exponents if snap_accepted else raw_exponents
        scale = snapped_scale if snap_accepted else raw_scale
        prediction = snapped_prediction if snap_accepted else raw_prediction
        factors = [
            f"({name}) ** ({float(power):.12g})"
            for (name, _), power in zip(features, exponents) if power != 0
        ]
        formula = f"({scale:.12g})" + (" * " + " * ".join(factors) if factors else "")
        evaluation = self.evaluate(
            f=self.parse_formula(formula), y=y_symbol,
            show_diagnostics=show_diagnostics,
        )
        evaluation["data_split_results"]["train"]["metrics"].update({
            "log_rmse": float(np.sqrt(np.mean((np.log(np.abs(prediction)) - log_y) ** 2))),
            "valid_sample_ratio": float(np.mean(valid)),
        })

        n_stability_subsets = min(max(int(n_stability_subsets), 1), 20)
        subset_exponents = []
        for repeat in range(n_stability_subsets):
            indices = np.random.default_rng(repeat).choice(
                len(log_y), max(2, int(0.8 * len(log_y))), replace=False
            )
            subset_design = (
                np.column_stack([np.ones(len(indices)), log_x[indices]])
                if include_scale else log_x[indices]
            )
            subset_coef = np.linalg.lstsq(subset_design, log_y[indices], rcond=None)[0]
            subset_exponents.append(subset_coef[1:] if include_scale else subset_coef)
        stability = np.std(np.asarray(subset_exponents), axis=0)

        result = evaluation | {
            "exponents": {name: float(value) for (name, _), value in zip(features, exponents)},
            "heuristic_exponent_std_across_subset_refits": {
                name: float(value) for (name, _), value in zip(features, stability)
            },
            "exceptions": [] if rank == design.shape[1] else ["Log-space design matrix is rank deficient."],
        }
        if snap_exponents:
            result["simple_exponent_check"] = {
                "rounding_applied": snap_accepted,
                "raw_exponents": {name: float(value) for (name, _), value in zip(features, raw_exponents)},
                "snapped_exponents": {name: float(value) for (name, _), value in zip(features, snapped_exponents)},
                "rmse_increase_as_fraction_of_target_rms": float(degradation),
            }
        excluded = {
            name: [float(np.min(values[~valid & np.isfinite(values)])), float(np.max(values[~valid & np.isfinite(values)]))]
            for name, values in features if np.any(~valid & np.isfinite(values))
        }
        if excluded:
            result["excluded_finite_ranges"] = excluded
        return result

    @staticmethod
    def _predict(features, valid, exponents, scale):
        return scale * np.prod(np.column_stack([
            values[valid] ** exponent
            for (_, values), exponent in zip(features, exponents)
        ]), axis=1)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        lines = [
            cls.format_evaluation_result(result, title="Best fitted power-law formula"),
            f"Training-set log-space RMSE={result['data_split_results']['train']['metrics']['log_rmse']:.6g}; "
            f"formula complexity={result['data_split_results']['train']['metrics']['complexity']}.",
            f"Samples satisfying the nonzero-target and positive-input requirements: "
            f"{result['data_split_results']['train']['metrics']['valid_sample_ratio']:.1%}.",
        ]
        exponent_parts = [
            f"{name}={value:.6g} (std {result['heuristic_exponent_std_across_subset_refits'][name]:.3g}; "
            f"{'stable' if result['heuristic_exponent_std_across_subset_refits'][name] <= max(0.05 * abs(value), 0.01) else 'unstable'})"
            for name, value in result["exponents"].items()
        ]
        lines.append(
            "Heuristic exponent stability from repeated fits on resampled data subsets: "
            + ", ".join(exponent_parts) + ". These standard deviations are stability diagnostics, "
            "not confidence intervals."
        )
        if check := result.get("simple_exponent_check"):
            change = check["rmse_increase_as_fraction_of_target_rms"]
            if check["rounding_applied"]:
                if abs(change) <= 1e-10:
                    effect = "caused no meaningful change in RMSE"
                else:
                    effect = f"changed RMSE by {change:.3g} of the target RMS (positive is worse)"
                lines.append(
                    f"Simple-exponent check: replaced {check['raw_exponents']} with "
                    f"{check['snapped_exponents']}; this {effect}, so the simplified exponents were used."
                )
            else:
                lines.append(
                    f"Simple-exponent check: proposed {check['snapped_exponents']} instead of "
                    f"{check['raw_exponents']}, but the simplification was rejected "
                    f"(RMSE change {change:.3g} of the target RMS; positive means worse fit)."
                )
        if "excluded_finite_ranges" in result:
            lines.append(
                "Domain warning: these finite input ranges were excluded because at least one power-law "
                f"domain requirement failed: {result['excluded_finite_ranges']}."
            )
        if result["exceptions"]:
            lines.append("Exceptions: " + "; ".join(result["exceptions"]))
        lines.append(
            "Eligible for submission (default target predicted without using the target as an input): "
            f"{result['is_candidate']}."
        )
        return "\n".join(lines)
