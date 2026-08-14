# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""变量关系、条件分布和一维 collapse 分析工具。"""
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from typing import Any, Dict, List
from .base_tool import BaseTool, ToolMetadata, is_numeric_array


@BaseTool.register("relationship_analysis")
class RelationshipAnalysisTool(BaseTool):
    metadata = ToolMetadata("relationship_analysis")

    def execute(
        self,
        variables: List[str] = None,
        y: str = None,
        n_bins: int = 10,
        pairwise: bool = False,
        binning: str = "quantile",
        validation_fraction: float = 0.2,
        n_repeats: int = 3,
        collapse_model: str = "bins",
    ) -> Dict[str, Any]:
        """Analyze feature-target relationships, conditional distributions, and one-dimensional collapse.

        Args:
            variables: Variables or expressions to analyze. Use all numeric non-target variables by default.
            y: Target variable or expression. Use the formula-discovery target by default.
            n_bins: Number of bins for each feature's conditional target summary (2-100).
            pairwise: Whether to also return full pairwise Pearson and Spearman matrices.
            binning: Binning strategy: "quantile" or "equal_width".
            validation_fraction: Fraction held out when estimating out-of-sample collapse (0.05-0.5).
            n_repeats: Number of deterministic holdout repeats used for collapse stability (1-20).
            collapse_model: One-dimensional probe: "bins", "spline", or "isotonic".
        """
        data = self.context["data"]
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        n_bins = max(2, min(int(n_bins), 100))
        validation_fraction = min(max(float(validation_fraction), 0.05), 0.5)
        n_repeats = min(max(int(n_repeats), 1), 20)
        if binning not in {"quantile", "equal_width"}:
            raise ValueError("binning must be 'quantile' or 'equal_width'")
        if collapse_model not in {"bins", "spline", "isotonic"}:
            raise ValueError("collapse_model must be 'bins', 'spline', or 'isotonic'")

        target_value = data[target_name] if target_name in data else self.parse_formula(target_name).eval(data)
        if not is_numeric_array(target_value):
            raise ValueError(f"Target '{target_name}' did not produce numeric values.")
        target = np.asarray(target_value, dtype=float).flatten()
        variables = variables or [
            name for name, value in data.items()
            if name != target_name and is_numeric_array(value)
        ]

        arrays = {}
        exceptions = []
        for expression in variables:
            try:
                value = data[expression] if expression in data else self.parse_formula(expression).eval(data)
                if not is_numeric_array(value):
                    raise ValueError("expression did not produce numeric values")
                value = np.asarray(value, dtype=float).flatten()
                if value.shape != target.shape:
                    raise ValueError(f"shape {value.shape} does not match target shape {target.shape}")
                arrays[expression] = value
            except Exception as exc:
                exceptions.append(f"Failed to compute '{expression}': {exc}")

        relationships = {}
        for name, values in arrays.items():
            try:
                relationship = self._analyze_relationship(
                    values, target, n_bins, binning,
                    validation_fraction, n_repeats, collapse_model,
                )

                # Check whether another variable still explains the cross-fitted residual.
                finite = np.isfinite(values) & np.isfinite(target)
                indices = np.flatnonzero(finite)
                if len(indices) >= 10:
                    prediction = np.full_like(target, np.nan, dtype=float)
                    folds = np.array_split(np.random.default_rng(0).permutation(indices), 5)
                    for test in folds:
                        train = np.setdiff1d(indices, test, assume_unique=False)
                        prediction[test] = self._fit_predict_1d(
                            values[train], target[train], values[test],
                            n_bins, binning, collapse_model,
                        )
                    residual = target - prediction
                    remaining = []
                    for other, other_values in arrays.items():
                        if other == name:
                            continue
                        pearson, spearman = self.correlation_coefficients(other_values, residual)
                        strength = max(abs(pearson), abs(spearman))
                        if np.isfinite(strength):
                            remaining.append({
                                "variable": other,
                                "pearson": pearson,
                                "spearman": spearman,
                                "strength": strength,
                            })
                    if remaining:
                        relationship["strongest_residual_association_after_one_variable_fit"] = max(
                            remaining, key=lambda item: item["strength"]
                        )
                relationships[name] = relationship
            except Exception as exc:
                exceptions.append(f"Failed to analyze '{name}': {exc}")

        result = {
            "target": target_name,
            "relationships": relationships,
            "exceptions": exceptions,
        }
        if pairwise:
            pairwise_arrays = {name: arrays[name] for name in relationships}
            if target_name not in pairwise_arrays:
                pairwise_arrays[target_name] = target
            names = list(pairwise_arrays)
            pearson_matrix, spearman_matrix = [], []
            for left in names:
                pearson_row, spearman_row = [], []
                for right in names:
                    pearson, spearman = self.correlation_coefficients(
                        pairwise_arrays[left], pairwise_arrays[right]
                    )
                    pearson_row.append(pearson)
                    spearman_row.append(spearman)
                pearson_matrix.append(pearson_row)
                spearman_matrix.append(spearman_row)
            result["pairwise_correlations"] = {
                "variables": names,
                "pearson": pearson_matrix,
                "spearman": spearman_matrix,
            }
        return result

    @classmethod
    def _analyze_relationship(
        cls, x, y, n_bins, binning, validation_fraction, n_repeats, collapse_model,
    ):
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if not len(x):
            raise ValueError("no jointly finite samples")

        if np.min(x) == np.max(x):
            edges = np.array([x[0], x[0]])
            bin_ids = np.zeros(len(x), dtype=int)
        else:
            edges = (
                np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
                if binning == "quantile"
                else np.linspace(np.min(x), np.max(x), n_bins + 1)
            )
            bin_ids = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

        conditional_bins = []
        for index in range(max(1, len(edges) - 1)):
            mask = bin_ids == index
            if not np.any(mask):
                continue
            xb, yb = x[mask], y[mask]
            y_mean = float(np.mean(yb))
            conditional_bins.append({
                "x_range": [float(np.min(xb)), float(np.max(xb))],
                "sample_count": int(len(yb)),
                "y_mean": y_mean,
                "y_std": float(np.std(yb)),
                "y_range": [float(np.min(yb)), float(np.max(yb))],
            })

        validation_scores = []
        if len(x) >= 10 and np.var(y) > 0:
            for repeat in range(n_repeats):
                order = np.random.default_rng(repeat).permutation(len(x))
                n_test = max(2, int(round(len(x) * validation_fraction)))
                test, train = order[:n_test], order[n_test:]
                if len(train) < 4:
                    continue
                try:
                    prediction = cls._fit_predict_1d(
                        x[train], y[train], x[test], n_bins, binning, collapse_model
                    )
                    baseline = float(np.sum((y[test] - np.mean(y[train])) ** 2))
                    if baseline > 0 and np.all(np.isfinite(prediction)):
                        validation_scores.append(float(
                            1 - np.sum((y[test] - prediction) ** 2) / baseline
                        ))
                except Exception:
                    continue

        pearson, spearman = cls.correlation_coefficients(x, y)
        bin_means = np.asarray([item["y_mean"] for item in conditional_bins])
        differences = np.diff(bin_means)
        if len(differences) and np.all(differences >= 0):
            shape = "increasing"
        elif len(differences) and np.all(differences <= 0):
            shape = "decreasing"
        elif len(bin_means) >= 3 and 0 < int(np.argmin(bin_means)) < len(bin_means) - 1:
            shape = "U-shaped"
        elif len(bin_means) >= 3 and 0 < int(np.argmax(bin_means)) < len(bin_means) - 1:
            shape = "inverted-U-shaped"
        else:
            shape = "non-monotonic or unclear"
        return {
            "pearson": pearson,
            "spearman": spearman,
            "observed_variable_range": [float(np.min(x)), float(np.max(x))],
            "one_variable_test_r2_mean": float(np.mean(validation_scores)) if validation_scores else float("nan"),
            "one_variable_test_r2_std": float(np.std(validation_scores)) if validation_scores else float("nan"),
            "one_variable_test_repeats": len(validation_scores),
            "binned_shape": shape,
            "conditional_bins": conditional_bins,
        }

    @staticmethod
    def _fit_predict_1d(train_x, train_y, test_x, n_bins, binning, model):
        order = np.argsort(train_x)
        x, y = train_x[order], train_y[order]
        if model == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            increasing = bool(stats.spearmanr(x, y).statistic >= 0)
            return IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, y).predict(test_x)
        if model == "spline":
            unique_x, inverse = np.unique(x, return_inverse=True)
            means = np.bincount(inverse, weights=y) / np.bincount(inverse)
            if len(unique_x) >= 4:
                smoothing = max(len(unique_x) * float(np.var(means)) * 1e-3, 0.0)
                return np.asarray(UnivariateSpline(
                    unique_x, means, k=min(3, len(unique_x) - 1), s=smoothing, ext=3
                )(test_x), dtype=float)
        if np.min(x) == np.max(x):
            return np.full_like(test_x, np.mean(y), dtype=float)
        edges = (
            np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
            if binning == "quantile"
            else np.linspace(np.min(x), np.max(x), n_bins + 1)
        )
        ids = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        global_mean = float(np.mean(y))
        means = np.asarray([
            np.mean(y[ids == index]) if np.any(ids == index) else global_mean
            for index in range(len(edges) - 1)
        ])
        test_ids = np.clip(np.digitize(test_x, edges[1:-1]), 0, len(means) - 1)
        return means[test_ids]

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        lines = [f"Target: {result['target']}"]
        for name, relationship in result["relationships"].items():
            lines.append(
                f"Relationship of {name} to {result['target']}: Pearson linear correlation="
                f"{relationship['pearson']:.4g}; Spearman rank correlation="
                f"{relationship['spearman']:.4g}."
            )
            lines.append(
                f"A flexible one-variable {relationship['one_variable_test_repeats']}-repeat held-out fit "
                f"using only {name} achieved mean test R²="
                f"{relationship['one_variable_test_r2_mean']:.4g} with R² standard deviation="
                f"{relationship['one_variable_test_r2_std']:.3g} across those data splits. A high value "
                f"means this flexible predictor can reproduce {result['target']} from {name} on the "
                f"observed range {relationship['observed_variable_range']}; this score does not identify "
                "the symbolic form and does not imply causality, necessity, or an independent contribution. "
                "Do not add these R² scores across variables because variables may encode the same "
                "information nonlinearly."
            )
            shape_hint = {
                "U-shaped": "Try even transforms such as a square or absolute value as hypotheses.",
                "inverted-U-shaped": "Try a negated even transform or another peaked nonlinear form as a hypothesis.",
                "increasing": "Try simple monotone transforms as hypotheses.",
                "decreasing": "Try inverse or decreasing monotone transforms as hypotheses.",
            }.get(relationship["binned_shape"], "Inspect the bins before choosing a transform.")
            lines.append(
                f"Heuristic shape clue from binned target means: {relationship['binned_shape']} "
                f"(sensitive to domain coverage and binning). {shape_hint}"
            )
            if strongest := relationship.get("strongest_residual_association_after_one_variable_fit"):
                lines.append(
                    f"In cross-fitted held-out predictions of {result['target']} from {name} alone, "
                    "the remaining errors were "
                    f"most associated with {strongest['variable']} (residual Pearson="
                    f"{strongest['pearson']:.4g}, residual Spearman={strongest['spearman']:.4g}). "
                    "This may indicate additional predictive information, not an independent causal effect; "
                    f"first check whether it is derived from or redundant with {name}."
                )
            for item in relationship["conditional_bins"]:
                lines.append(
                    f"  For {name} in {item['x_range']} (n={item['sample_count']}), "
                    f"{result['target']} mean={item['y_mean']:.4g}, standard deviation="
                    f"{item['y_std']:.4g}, range={item['y_range']}."
                )
        if matrix := result.get("pairwise_correlations"):
            lines.append(
                "Marginal pairwise linear/rank associations (not conditioned and not causal; near-zero "
                "values do not rule out nonlinear dependence):"
            )
            for left in range(len(matrix["variables"])):
                for right in range(left + 1, len(matrix["variables"])):
                    lines.append(
                        f"  {matrix['variables'][left]} ↔ {matrix['variables'][right]}: "
                        f"Pearson={matrix['pearson'][left][right]:.4g}, "
                        f"Spearman={matrix['spearman'][left][right]:.4g}."
                    )
        if result["exceptions"]:
            lines.append("Exceptions:\n" + "\n".join(result["exceptions"]))
        return "\n".join(lines)
