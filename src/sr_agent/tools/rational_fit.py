# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""有理式拟合工具。"""
import numpy as np
from typing import Any, Dict, List
from .polynomial_fit import PolynomialFitTool
from .base_tool import BaseTool, ToolMetadata, is_numeric_array


@BaseTool.register("rational_fit")
class RationalFitTool(BaseTool):
    metadata = ToolMetadata("rational_fit")

    def execute(
        self,
        x: List[str] = None,
        y: str = None,
        numerator_degree: int = 2,
        denominator_degree: int = 1,
        include_interactions: bool = True,
        numerator_degrees: List[int] = None,
        denominator_degrees: List[int] = None,
        validation_fraction: float = 0.2,
        top_k: int = 5,
        complexity_penalty: float = 1e-12,
        n_stability_subsets: int = 5,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Fit a rational expression P(x) / Q(x) by linearized least squares with Q's constant fixed to one.

        Args:
            x: Input variables or expressions. Use all numeric non-target variables by default.
            y: Target variable or expression. Use the formula-discovery target by default.
            numerator_degree: Maximum total degree of numerator P (0-8).
            denominator_degree: Maximum total degree of denominator Q excluding its fixed constant (0-8).
            include_interactions: Whether polynomial terms may contain multiple input features.
            numerator_degrees: Optional numerator degree grid; overrides numerator_degree when provided.
            denominator_degrees: Optional denominator degree grid; overrides denominator_degree when provided.
            validation_fraction: Deterministic holdout fraction used to rank degree combinations (0.05-0.5).
            top_k: Number of degree-grid candidates to return (1-20).
            complexity_penalty: Penalty per fitted coefficient added to validation RMSE after target-scale normalization.
            n_stability_subsets: Number of deterministic 80% subsets used to measure selected-model stability (1-20).
            show_diagnostics: Whether final metrics should include compact residual diagnostics.
        """
        data = self.context["data"]
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        x = x or [name for name, value in data.items() if name != target_name and is_numeric_array(value)]
        p_degrees = sorted(set(
            max(0, min(int(value), 8))
            for value in (numerator_degrees if numerator_degrees is not None else [numerator_degree])
        ))
        q_degrees = sorted(set(
            max(0, min(int(value), 8))
            for value in (denominator_degrees if denominator_degrees is not None else [denominator_degree])
        ))
        validation_fraction = min(max(float(validation_fraction), 0.05), 0.5)
        top_k = min(max(int(top_k), 1), 20)
        complexity_penalty = max(float(complexity_penalty), 0.0)
        n_stability_subsets = min(max(int(n_stability_subsets), 1), 20)
        eq_y = self.parse_formula(target_name)
        target = np.asarray(eq_y.eval(data), dtype=float).flatten()
        symbols, exceptions = [], []
        for expression in x:
            try:
                symbol = self.parse_formula(expression)
                value = np.asarray(symbol.eval(data), dtype=float).flatten()
                if value.shape != target.shape:
                    raise ValueError("shape mismatch")
                symbols.append(symbol)
            except Exception as exc:
                exceptions.append(f"Failed to compute '{expression}': {exc}")
        if not symbols:
            raise ValueError("No valid input variables available for fitting.")
        helper = PolynomialFitTool(data=data, target=self.context["target"])
        allowed = helper._get_allowed_interactions(symbols, include_interactions, None, None)
        finite_target = np.isfinite(target)
        finite_indices = np.flatnonzero(finite_target)
        order = np.random.default_rng(0).permutation(finite_indices)
        n_validation = max(2, int(round(len(order) * validation_fraction)))
        validation_indices, train_indices = order[:n_validation], order[n_validation:]
        candidates = []
        for p_degree in p_degrees:
            for q_degree in q_degrees:
                try:
                    model = self._fit_model(helper, symbols, allowed, data, target, train_indices, p_degree, q_degree)
                    candidate = self._score_model(
                        model, target, eq_y, train_indices, validation_indices,
                        show_diagnostics=False,
                    )
                    candidate["numerator_degree"] = p_degree
                    candidate["denominator_degree"] = q_degree
                    candidates.append(candidate)
                except Exception as exc:
                    exceptions.append(f"Degree ({p_degree}, {q_degree}) failed: {exc}")
        if not candidates:
            raise ValueError("No rational degree combination could be fitted.")
        target_scale = float(np.sqrt(np.mean(target[finite_target] ** 2)))
        for candidate in candidates:
            candidate["selection_score"] = (
                candidate["validation_rmse"] / max(target_scale, np.finfo(float).eps)
                + complexity_penalty * candidate["complexity"]
            )
        for candidate in candidates:
            candidate["pareto_optimal"] = not any(
                other["complexity"] <= candidate["complexity"]
                and other["validation_rmse"] <= candidate["validation_rmse"]
                and (other["complexity"] < candidate["complexity"] or other["validation_rmse"] < candidate["validation_rmse"])
                for other in candidates
            )
        candidates.sort(key=lambda item: (item["selection_score"], item["complexity"]))
        selected = candidates[0]
        selected_model = self._fit_model(
            helper, symbols, allowed, data, target, finite_indices,
            selected["numerator_degree"], selected["denominator_degree"],
        )
        final = self._score_model(
            selected_model, target, eq_y, finite_indices, finite_indices,
            show_diagnostics=show_diagnostics,
        )
        stability_rmse = []
        for repeat in range(n_stability_subsets):
            order = np.random.default_rng(repeat).permutation(finite_indices)
            cut = max(2, int(0.8 * len(order)))
            subset_train, subset_validation = order[:cut], order[cut:]
            if not len(subset_validation):
                continue
            try:
                subset_model = self._fit_model(
                    helper, symbols, allowed, data, target, subset_train,
                    selected["numerator_degree"], selected["denominator_degree"],
                )
                subset_result = self._score_model(
                    subset_model, target, eq_y, subset_train, subset_validation,
                    show_diagnostics=False,
                )
                stability_rmse.append(subset_result["validation_rmse"])
            except Exception as exc:
                exceptions.append(f"Stability subset {repeat + 1} failed: {exc}")

        alternatives = []
        for candidate in candidates:
            if (
                candidate is selected
                or not candidate["pareto_optimal"]
                or candidate["validation_rmse"] > selected["validation_rmse"] + 0.05 * target_scale
            ):
                continue
            alternatives.append({
                "formula": candidate["formula"],
                "numerator_degree": candidate["numerator_degree"],
                "denominator_degree": candidate["denominator_degree"],
                "validation_rmse": candidate["validation_rmse"],
                "formula_complexity": candidate["complexity"],
            })
            if len(alternatives) >= top_k:
                break
        return final["evaluation"] | {
            "metrics": final["evaluation"]["metrics"] | {
                "valid_sample_ratio": final["valid_sample_ratio"],
            },
            "selected_polynomial_degrees": {
                "numerator_degree": selected["numerator_degree"],
                "denominator_degree": selected["denominator_degree"],
            },
            "alternatives": alternatives,
            "denominator_safety_on_observed_samples": final["denominator_diagnostics"],
            "heldout_rmse_across_subsamples": {
                "number_of_subsamples": len(stability_rmse),
                "mean_rmse": float(np.mean(stability_rmse)) if stability_rmse else float("nan"),
                "rmse_std": float(np.std(stability_rmse)) if stability_rmse else float("nan"),
            },
            "exceptions": exceptions + ([] if final["matrix_rank"] == final["complexity"] else ["Linearized design matrix is rank deficient."]),
        }

    def _fit_model(self, helper, symbols, allowed, data, target, fit_indices, p_degree, q_degree):
        p_terms = helper.generate_terms(symbols, p_degree, allowed, True)
        q_terms = [term for term in helper.generate_terms(symbols, q_degree, allowed, True) if term.to_str() != "1"]
        p_matrix = helper._build_design_matrix(data, p_terms, len(target))
        q_matrix = helper._build_design_matrix(data, q_terms, len(target)) if q_terms else np.empty((len(target), 0))
        design = np.column_stack([p_matrix, -target[:, None] * q_matrix])
        fit_indices = np.asarray(fit_indices, dtype=int)
        usable = fit_indices[np.all(np.isfinite(design[fit_indices]), axis=1) & np.isfinite(target[fit_indices])]
        if len(usable) < design.shape[1]:
            raise ValueError(f"need at least {design.shape[1]} finite fitting samples, got {len(usable)}")
        coefficients, _, rank, _ = np.linalg.lstsq(design[usable], target[usable], rcond=None)
        return {
            "p_terms": p_terms, "q_terms": q_terms, "p_matrix": p_matrix, "q_matrix": q_matrix,
            "p_coef": coefficients[:len(p_terms)], "q_coef": coefficients[len(p_terms):],
            "rank": int(rank),
            "complexity": int(design.shape[1]),
        }

    def _score_model(
        self, model, target, y_symbol, train_indices, validation_indices,
        show_diagnostics: bool = False,
    ):
        denominator = 1 + model["q_matrix"] @ model["q_coef"]
        prediction = np.full_like(target, np.nan, dtype=float)
        safe = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        prediction[safe] = (model["p_matrix"][safe] @ model["p_coef"]) / denominator[safe]
        train = np.asarray(train_indices, dtype=int)
        validation = np.asarray(validation_indices, dtype=int)
        train = train[np.isfinite(prediction[train]) & np.isfinite(target[train])]
        validation = validation[np.isfinite(prediction[validation]) & np.isfinite(target[validation])]
        if not len(train) or not len(validation):
            raise ValueError("model has no finite train or validation predictions")
        numerator_parts = [
            f"({float(coef):.12g}) * ({term.to_str()})"
            for coef, term in zip(model["p_coef"], model["p_terms"]) if coef != 0
        ]
        numerator = " + ".join(numerator_parts) if numerator_parts else "0"
        denominator_formula = "1" + "".join(
            f" + ({float(coef):.12g}) * ({term.to_str()})"
            for coef, term in zip(model["q_coef"], model["q_terms"]) if coef != 0
        )
        formula = f"({numerator}) / ({denominator_formula})"
        formula_symbol = self.parse_formula(formula)
        abs_den = np.abs(denominator[np.isfinite(denominator)])
        quantiles = np.quantile(abs_den, [0, 0.01]).tolist() if len(abs_den) else [float("nan")] * 2
        validation_evaluation = self.evaluate(
            f=formula_symbol, y=y_symbol, y_pred=prediction[validation], y_true=target[validation],
            show_diagnostics=show_diagnostics,
        )
        return {
            "formula": formula,
            "validation_rmse": validation_evaluation["metrics"]["rmse"],
            "evaluation": validation_evaluation,
            "complexity": model["complexity"], "matrix_rank": model["rank"],
            "valid_sample_ratio": float(np.mean(np.isfinite(prediction))),
            "denominator_diagnostics": {
                "minimum_absolute_denominator": float(quantiles[0]),
                "first_percentile_absolute_denominator": float(quantiles[1]),
            },
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        degrees = result["selected_polynomial_degrees"]
        safety = result["denominator_safety_on_observed_samples"]
        stability = result["heldout_rmse_across_subsamples"]
        lines = [
            f"Best fitted rational formula: {result['formula']}",
            f"Selected structure: numerator polynomial degree {degrees['numerator_degree']}; "
            f"denominator polynomial degree {degrees['denominator_degree']}.",
            f"Fit quality on all usable samples: RMSE={result['metrics']['rmse']:.6g}, "
            f"R²={result['metrics']['r2']:.6g}, formula complexity={result['metrics']['complexity']}.",
            f"Denominator safety on observed samples: minimum |denominator|="
            f"{safety['minimum_absolute_denominator']:.6g}; 1st percentile="
            f"{safety['first_percentile_absolute_denominator']:.6g}. "
            "Values near zero indicate a possible pole; this check does not guarantee safety outside "
            "the observed input range.",
            f"Held-out RMSE across {stability['number_of_subsamples']} repeated fits on resampled data subsets: "
            f"mean RMSE={stability['mean_rmse']:.6g}, RMSE standard deviation="
            f"{stability['rmse_std']:.3g} (a large standard deviation means performance depends on the split).",
        ]
        if result["alternatives"]:
            lines.append("Simpler non-dominated alternatives (validation error versus complexity):")
            for alternative in result["alternatives"]:
                lines.append(
                    f"  - {alternative['formula']} | validation RMSE="
                    f"{alternative['validation_rmse']:.6g}, formula complexity="
                    f"{alternative['formula_complexity']}."
                )
        if result["exceptions"]:
            lines.append("Exceptions: " + "; ".join(result["exceptions"]))
        lines.append(
            "Eligible for submission (default target predicted without using the target as an input): "
            f"{result['is_candidate']}."
        )
        return "\n".join(lines)
