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
        eq_y = self.parse_formula(target_name)
        target = np.asarray(eq_y.eval(data), dtype=float).flatten()
        symbols, valid_x, exceptions = [], [], []
        for expression in x:
            try:
                symbol = self.parse_formula(expression)
                value = np.asarray(symbol.eval(data), dtype=float).flatten()
                if value.shape != target.shape:
                    raise ValueError("shape mismatch")
                symbols.append(symbol)
                valid_x.append(expression)
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
        diagnostics = final["evaluation"]["data_split_results"]["train"].get("diagnostics")
        if diagnostics:
            for expression, symbol in zip(valid_x, symbols):
                values = np.asarray(symbol.eval(data), dtype=float).flatten()
                for sample in diagnostics["worst_samples"]:
                    sample["row"][expression] = float(values[sample["index"]])
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
            "analyzed_input_expressions": valid_x,
            "data_split_results": final["evaluation"]["data_split_results"] | {
                "train": final["evaluation"]["data_split_results"]["train"] | {
                    "metrics": final["evaluation"]["data_split_results"]["train"]["metrics"] | {
                        "valid_sample_ratio": final["valid_sample_ratio"],
                    },
                },
            },
            "selected_polynomial_degrees": {
                "numerator_degree": selected["numerator_degree"],
                "denominator_degree": selected["denominator_degree"],
            },
            "alternatives": alternatives,
            "denominator_safety_on_observed_samples": final["denominator_diagnostics"],
            "linearized_design_matrix": {
                "rows": final["matrix_rows"],
                "columns": final["complexity"],
                "rank": final["matrix_rank"],
                "column_labels": final["matrix_column_labels"],
            },
            "exceptions": exceptions,
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
            "rows": int(len(usable)),
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
        denominator_expression = self.parse_formula(denominator_formula).to_str()
        target_label = y_symbol.to_str()

        def factor_label(expression: str) -> str:
            return expression if expression.isidentifier() else f"({expression})"

        matrix_column_labels = [term.to_str() for term in model["p_terms"]] + [
            f"-{factor_label(target_label)}*{factor_label(term.to_str())}"
            for term in model["q_terms"]
        ]
        formula = f"({numerator}) / ({denominator_formula})"
        formula_symbol = self.parse_formula(formula)
        finite_denominator = denominator[np.isfinite(denominator)]
        abs_den = np.abs(finite_denominator)
        denominator_min = float(np.min(finite_denominator)) if finite_denominator.size else float("nan")
        denominator_max = float(np.max(finite_denominator)) if finite_denominator.size else float("nan")
        thresholds = (1e-8, 1e-6, 1e-4, 1e-2)
        validation_metrics = self.calculate_metrics(
            formula_symbol, target[validation], prediction[validation]
        )
        validation_evaluation = self.evaluate(
            f=formula_symbol, y=y_symbol,
            show_diagnostics=show_diagnostics,
        )
        return {
            "formula": formula,
            "validation_rmse": validation_metrics["rmse"],
            "evaluation": validation_evaluation,
            "complexity": model["complexity"], "matrix_rank": model["rank"],
            "matrix_rows": model["rows"],
            "matrix_column_labels": matrix_column_labels,
            "valid_sample_ratio": float(np.mean(np.isfinite(prediction))),
            "denominator_diagnostics": {
                "formula": denominator_expression,
                "minimum": denominator_min,
                "maximum": denominator_max,
                "crosses_zero": bool(denominator_min <= 0 <= denominator_max),
                "finite_samples": int(finite_denominator.size),
                "total_samples": int(denominator.size),
                "near_zero_counts": {
                    threshold: int(np.count_nonzero(abs_den < threshold))
                    for threshold in thresholds
                },
            },
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        lines = [cls.format_evaluation_result(result, title="Best fitted rational formula")]
        if result["alternatives"]:
            lines.append("Simpler non-dominated alternatives (validation error versus complexity):")
            for alternative in result["alternatives"]:
                lines.append(
                    f"  - {alternative['formula']} | validation RMSE="
                    f"{alternative['validation_rmse']:.6g}, formula complexity="
                    f"{alternative['formula_complexity']}."
                )
        exception_lines = []
        matrix = result["linearized_design_matrix"]
        if matrix["rank"] < matrix["columns"]:
            labels = ", ".join(matrix["column_labels"])
            exception_lines.extend([
                (1, "Linearized design matrix is rank deficient."),
                (2, f"Rank[{labels}] = {matrix['rank']} < {matrix['columns']}."),
            ])
        denominator = result["denominator_safety_on_observed_samples"]
        if denominator["crosses_zero"]:
            minimum = f"{denominator['minimum']:#.3g}".removesuffix(".")
            maximum = f"{denominator['maximum']:#.3g}".removesuffix(".")
            exception_lines.append(
                (1, f"Denominator ({denominator['formula']}) crosses or reaches zero "
                    f"on training samples: range=[{minimum}, {maximum}].")
            )
            n_finite = denominator["finite_samples"]
            for threshold, count in denominator["near_zero_counts"].items():
                fraction = count / n_finite if n_finite else float("nan")
                displayed_percent = f"{100 * fraction:#.3g}".removesuffix(".")
                threshold_label = f"1e-{int(round(-np.log10(threshold)))}"
                exception_lines.append(
                    (2, f"fraction(|denominator| < {threshold_label})="
                     f"{count}/{n_finite} ({displayed_percent}%).")
                )
        exception_lines.extend((1, message) for message in result["exceptions"])
        if exception_lines:
            lines.append("Exception:")
            lines.extend(f"{'    ' * depth}{message}" for depth, message in exception_lines)
        return "\n".join(lines)
