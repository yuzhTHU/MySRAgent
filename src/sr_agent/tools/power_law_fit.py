# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""乘法幂律拟合工具。"""
import numpy as np
from fractions import Fraction
from scipy.stats import t as student_t
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
        n_stability_folds: int = 5,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Fit a multiplicative power law y = c * product(x_i ** p_i) in log space.

        Args:
            x: Strictly positive input variables or expressions on every training sample. Use all numeric non-target variables by default.
            y: Strictly positive target variable or expression on every training sample. Use the formula-discovery target by default.
            include_scale: Whether to fit the multiplicative scale c.
            snap_exponents: Whether to try nearby simple rational exponents and refit c.
            max_denominator: Largest denominator considered when snapping exponents (1-32).
            snap_tolerance: Maximum distance from every fitted exponent to its snapped value.
            max_snap_rmse_degradation: Maximum normalized RMSE degradation allowed after snapping.
            n_stability_folds: Number of disjoint folds for leave-one-fold-out exponent confidence intervals (2-20).
            show_diagnostics: Whether final metrics should include compact residual diagnostics.
        """
        data = self.context["data"]
        if y is not None and not isinstance(y, str):
            return {"exceptions": [f"y must be a string expression, got {type(y).__name__}."]}
        if x is not None and (not isinstance(x, list) or any(not isinstance(item, str) for item in x)):
            return {"exceptions": ["x must be a list of string expressions."]}
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        x = x or [name for name, value in data.items() if name != target_name and is_numeric_array(value)]
        y_symbol = self.parse_formula(target_name)
        target = np.asarray(y_symbol.eval(data), dtype=float).flatten()
        if target.size == 0:
            return {"exceptions": [f"{target_name}: no training values; power_law_fit requires positive samples."]}

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

        exceptions = []
        for name, values in [*features, (target_name, target)]:
            finite = np.isfinite(values)
            nonpositive = int(np.count_nonzero(finite & (values <= 0)))
            nonfinite = int(np.count_nonzero(~finite))
            total = values.size
            if nonpositive:
                exceptions.append(
                    f"{name} has {nonpositive}/{total} "
                    f"({self._format_percentage(nonpositive, total)}) non-positive finite values"
                )
            if nonfinite:
                exceptions.append(
                    f"{name} has {nonfinite}/{total} "
                    f"({self._format_percentage(nonfinite, total)}) non-finite values"
                )
        if exceptions:
            return {"exceptions": exceptions, "domain_inapplicable": True}
        if len(target) < 2:
            return {"exceptions": ["At least two training samples are required for exponent confidence intervals."]}

        valid = np.ones(target.shape, dtype=bool)

        log_x = np.column_stack([np.log(values[valid]) for _, values in features])
        log_y = np.log(np.abs(target[valid]))
        design = np.column_stack([np.ones(len(log_x)), log_x]) if include_scale else log_x
        coefficients, _, rank, _ = np.linalg.lstsq(design, log_y, rcond=None)
        if include_scale:
            log_scale, raw_exponents = coefficients[0], coefficients[1:]
        else:
            log_scale, raw_exponents = 0.0, coefficients
        raw_scale = float(np.exp(log_scale))

        max_denominator = max(1, min(int(max_denominator), 32))
        snapped_fractions = [
            Fraction(float(value)).limit_denominator(max_denominator)
            for value in raw_exponents
        ]
        snapped_exponents = np.asarray([float(value) for value in snapped_fractions])
        snapped_log_scale = (
            float(np.mean(log_y - log_x @ snapped_exponents)) if include_scale else 0.0
        )
        snapped_scale = float(np.exp(snapped_log_scale))

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
        factors = []
        for index, ((name, _), power) in enumerate(zip(features, exponents)):
            if power == 0:
                continue
            if snap_accepted:
                fraction = snapped_fractions[index]
                exponent = (
                    f"Number({fraction.numerator}) / Number({fraction.denominator})"
                    if fraction.denominator != 1 else str(fraction.numerator)
                )
            else:
                exponent = f"{float(power):.12g}"
            base = name if name.isidentifier() else f"({name})"
            exponent = f"({exponent})" if "/" in exponent else exponent
            factors.append(f"{base} ** {exponent}")
        formula = f"{scale:.12g}" + (" * " + " * ".join(factors) if factors else "")
        evaluation = self.evaluate(
            f=self.parse_formula(formula), y=y_symbol,
            show_diagnostics=show_diagnostics,
        )
        evaluation["data_split_results"]["train"]["metrics"].update({
            "log_rmse": float(np.sqrt(np.mean((np.log(np.abs(prediction)) - log_y) ** 2))),
            "valid_sample_ratio": float(np.mean(valid)),
        })

        n_stability_folds = min(max(int(n_stability_folds), 2), 20, len(log_y))
        folds = np.array_split(np.random.default_rng(0).permutation(len(log_y)), n_stability_folds)
        leave_fold_out_exponents = []
        for held_out in range(n_stability_folds):
            fit_indices = np.concatenate([fold for i, fold in enumerate(folds) if i != held_out])
            subset_coef = np.linalg.lstsq(design[fit_indices], log_y[fit_indices], rcond=None)[0]
            leave_fold_out_exponents.append(subset_coef[1:] if include_scale else subset_coef)
        leave_fold_out_exponents = np.asarray(leave_fold_out_exponents)
        pseudo_values = (
            n_stability_folds * raw_exponents
            - (n_stability_folds - 1) * leave_fold_out_exponents
        )
        standard_errors = np.std(pseudo_values, axis=0, ddof=1) / np.sqrt(n_stability_folds)
        margin = student_t.ppf(0.975, df=n_stability_folds - 1) * standard_errors

        result = evaluation | {
            "exponents": {name: float(value) for (name, _), value in zip(features, exponents)},
            "exponent_confidence_intervals": {
                name: [float(value - width), float(value + width)]
                for (name, _), value, width in zip(features, raw_exponents, margin)
            },
            "exponent_stability_folds": {
                "number_of_folds": n_stability_folds,
                "training_samples": len(log_y),
                "method": "delete-one-fold jackknife t interval",
                "degrees_of_freedom": n_stability_folds - 1,
            },
            "exceptions": [] if rank == design.shape[1] else ["Log-space design matrix is rank deficient."],
        }
        if snap_exponents:
            raw_validation_r2 = self._validation_r2(features, y_symbol, raw_exponents, raw_scale)
            snapped_validation_r2 = self._validation_r2(
                features, y_symbol, snapped_exponents, snapped_scale
            )
            result["simple_exponent_check"] = {
                "rounding_applied": snap_accepted,
                "raw_exponents": {name: float(value) for (name, _), value in zip(features, raw_exponents)},
                "snapped_exponents": {name: float(value) for (name, _), value in zip(features, snapped_exponents)},
                "snapped_exponent_expressions": {
                    name: str(value) for (name, _), value in zip(features, snapped_fractions)
                },
                "rmse_increase_as_fraction_of_target_rms": float(degradation),
                "training_samples": len(log_y),
                "raw_validation_r2": raw_validation_r2,
                "snapped_validation_r2": snapped_validation_r2,
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

    @staticmethod
    def _format_percentage(count: int, total: int) -> str:
        value = 100 * count / total
        display = f"{value:.3g}"
        if "." not in display and "e" not in display:
            display += ".0"
        return f"{display}%"

    def _validation_r2(self, features, y_symbol, exponents, scale):
        """Score an exponent candidate on the same complete, positive validation rows."""
        data = self.context.get("evaluation_data")
        if not data:
            return None
        try:
            target = np.asarray(y_symbol.eval(data), dtype=float).flatten()
            validation_features = []
            for expression, _ in features:
                values = data[expression] if expression in data else self.parse_formula(expression).eval(data)
                values = np.asarray(values, dtype=float).flatten()
                if values.shape != target.shape or not np.all(np.isfinite(values) & (values > 0)):
                    return None
                validation_features.append((expression, values))
            if not np.all(np.isfinite(target) & (target > 0)):
                return None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                prediction = self._predict(
                    validation_features, np.ones(target.shape, dtype=bool), exponents, scale
                )
            if not np.all(np.isfinite(prediction)):
                return None
            total = float(np.sum((target - np.mean(target)) ** 2))
            return float(1 - np.sum((prediction - target) ** 2) / total) if total > 0 else None
        except (ArithmeticError, TypeError, ValueError, KeyError):
            return None

    @staticmethod
    def _format_ci_endpoints(lower: float, upper: float) -> tuple[str, str]:
        """Show at least three significant digits, up to six if needed to separate endpoints."""
        for precision in range(3, 7):
            left = format(lower, f"#.{precision}g").removesuffix(".")
            right = format(upper, f"#.{precision}g").removesuffix(".")
            if left != right or precision == 6:
                return left, right

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if "data_split_results" not in result:
            if result.get("domain_inapplicable"):
                issues = "\n".join(f"    {issue}" for issue in result["exceptions"])
                return (
                    f"Power-law fitting not applicable:\n{issues}\n"
                    "power_law_fit requires every x and y value to be finite and strictly positive."
                )
            return "Power-law fitting not applicable: " + "; ".join(result.get("exceptions", []))
        lines = [cls.format_evaluation_result(result, title="Best fitted power-law formula")]
        refits = result["exponent_stability_folds"]
        n_folds = refits["number_of_folds"]
        lines.append(
            f"Exponent stability (Divide the {refits['training_samples']} training samples into "
            f"{n_folds} folds, use {n_folds - 1} of the folds to fit the exponents, "
            f"and repeat the process {n_folds} times):"
        )
        for name, (lower, upper) in result["exponent_confidence_intervals"].items():
            left, right = cls._format_ci_endpoints(lower, upper)
            lines.append(f"    {name}: 95% CI=[{left}, {right}];")
        if check := result.get("simple_exponent_check"):
            lines.append("Simplified exponent check:")
            snapped_display = check.get("snapped_exponent_expressions", check["snapped_exponents"])
            raw_r2 = check["raw_validation_r2"]
            snapped_r2 = check["snapped_validation_r2"]
            if raw_r2 is None or snapped_r2 is None:
                effect = (
                    f"replacing them with {snapped_display} cannot be compared "
                    "on a validation set"
                )
            else:
                change = snapped_r2 - raw_r2
                magnitude = f"{abs(change):#.3g}".removesuffix(".")
                if change > 0:
                    verb = "increased" if check["rounding_applied"] else "would increase"
                    effect = f"replacing them with {snapped_display} {verb} validation-set R2 by {magnitude}"
                elif change < 0:
                    verb = "decreased" if check["rounding_applied"] else "would decrease"
                    effect = f"replacing them with {snapped_display} {verb} validation-set R2 by {magnitude}"
                else:
                    verb = "did not change" if check["rounding_applied"] else "would not change"
                    effect = f"replacing them with {snapped_display} {verb} validation-set R2"
            ending = (
                "; the simplified exponents were thus used as the best fitted power-law formula."
                if check["rounding_applied"] else "."
            )
            lines.append(
                f"    Direct fitting on {check['training_samples']} samples yields exponents: "
                f"{check['raw_exponents']}; {effect}{ending}"
            )
        if "excluded_finite_ranges" in result:
            lines.append(
                "Domain warning: these finite input ranges were excluded because at least one power-law "
                f"domain requirement failed: {result['excluded_finite_ranges']}."
            )
        if result["exceptions"]:
            lines.append("Exceptions: " + "; ".join(result["exceptions"]))
        return "\n".join(lines)
