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
        n_bins: int = 5,
        pairwise: bool = False,
        binning: str = "quantile",
        n_folds: int = 5,
        collapse_model: str = "bins",
    ) -> Dict[str, Any]:
        """Analyze feature-target relationships, conditional distributions, and one-dimensional collapse.

        Args:
            variables: Variables or expressions to analyze. Use all numeric non-target variables by default.
            y: Target variable or expression. Use the formula-discovery target by default.
            n_bins: Number of bins for each feature's conditional target summary (2-100).
            pairwise: Whether to also return full pairwise Pearson and Spearman matrices.
            binning: Binning strategy: "quantile" or "equal_width".
            n_folds: Number of disjoint cross-validation folds (2-20). Default: 5.
            collapse_model: One-dimensional predictor fitted on each training fold: "bins", "spline", or "isotonic".
        """
        data = self.context["data"]
        target_name = (y or self.context["target"]).strip().strip('"').strip("'")
        n_bins = max(2, min(int(n_bins), 100))
        n_folds = max(2, min(int(n_folds), 20))
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
                relationship, residual = self._analyze_relationship(
                    values, target, n_bins, binning, n_folds, collapse_model,
                )

                # Compute correlations of other variables with the cross-fitted residual.
                if np.any(np.isfinite(residual)):
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
            "analysis_settings": {
                "n_bins": n_bins,
                "binning": binning,
                "n_folds": n_folds,
                "collapse_model": collapse_model,
            },
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
        cls, x, y, n_bins, binning, n_folds, collapse_model,
    ):
        finite = np.isfinite(x) & np.isfinite(y)
        total_count = len(x)
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
                "bin_interval": [float(edges[index]), float(edges[index + 1])],
                "upper_inclusive": index == len(edges) - 2,
                "x_range": [float(np.min(xb)), float(np.max(xb))],
                "sample_count": int(len(yb)),
                "y_mean": y_mean,
                "y_std": float(np.std(yb)),
                "y_range": [float(np.min(yb)), float(np.max(yb))],
            })

        fold_scores = []
        prediction = np.full(len(x), np.nan, dtype=float)
        if len(x) >= n_folds and np.var(y) > 0:
            folds = np.array_split(np.random.default_rng(0).permutation(len(x)), n_folds)
            for test in folds:
                train_mask = np.ones(len(x), dtype=bool)
                train_mask[test] = False
                train = np.flatnonzero(train_mask)
                if len(train) < 4 or len(test) < 2:
                    continue
                try:
                    predicted = cls._fit_predict_1d(
                        x[train], y[train], x[test], n_bins, binning, collapse_model
                    )
                    baseline = float(np.sum((y[test] - np.mean(y[train])) ** 2))
                    if baseline > 0 and np.all(np.isfinite(predicted)):
                        prediction[test] = predicted
                        fold_scores.append(float(
                            1 - np.sum((y[test] - predicted) ** 2) / baseline
                        ))
                except Exception:
                    continue

        pearson, spearman = cls.correlation_coefficients(x, y)
        residual = np.full(total_count, np.nan, dtype=float)
        residual[finite] = y - prediction
        return {
            "pearson": pearson,
            "spearman": spearman,
            "jointly_finite_sample_count": len(x),
            "total_sample_count": total_count,
            "observed_variable_range": [float(np.min(x)), float(np.max(x))],
            "observed_target_range": [float(np.min(y)), float(np.max(y))],
            "one_variable_test_r2_mean": float(np.mean(fold_scores)) if fold_scores else float("nan"),
            "one_variable_test_r2_std": float(np.std(fold_scores)) if fold_scores else float("nan"),
            "one_variable_test_success_folds": len(fold_scores),
            "conditional_bins": conditional_bins,
        }, residual

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
        def number(value):
            if value == 0:
                return "0"
            return f"{value:#.3g}".removesuffix(".")

        def interval(values):
            return f"[{number(values[0])}, {number(values[1])}]"

        lines = []
        settings = result["analysis_settings"]
        predictor = {
            "bins": (
                f"Randomly split data into n_folds={settings['n_folds']} parts, "
                f"use {settings['n_folds'] - 1} parts for binning, and predict each "
                "held-out value using the bin mean"
            ),
            "spline": (
                f"Randomly split data into n_folds={settings['n_folds']} parts, "
                f"use {settings['n_folds'] - 1} parts to fit a univariate smoothing "
                "spline, and predict the remaining part"
            ),
            "isotonic": (
                f"Randomly split data into n_folds={settings['n_folds']} parts, "
                f"use {settings['n_folds'] - 1} parts to fit isotonic regression, "
                "and predict the remaining part"
            ),
        }[settings["collapse_model"]]
        duplicate_note = "; duplicate quantile cut points removed" if settings["binning"] == "quantile" else ""
        for name, relationship in result["relationships"].items():
            lines.extend([
                f"{name} vs {result['target']} (Only jointly finite feature/target samples are used, "
                f"{relationship['jointly_finite_sample_count']}/{relationship['total_sample_count']} finite samples):",
                f"  Pearson linear correlation={number(relationship['pearson'])};",
                f"  Spearman rank correlation={number(relationship['spearman'])};",
                f"  {name} range={interval(relationship['observed_variable_range'])};",
                f"  {result['target']} range={interval(relationship['observed_target_range'])};",
                f"  Binning analysis (n_bins={settings['n_bins']}, "
                f"method={settings['binning']}{duplicate_note}):",
                f"    (Range of {name} | samples | {result['target']} mean | "
                f"{result['target']} std | {result['target']} range)",
            ])
            for item in relationship["conditional_bins"]:
                lower, upper = item["bin_interval"]
                if lower == upper:
                    condition = f"{name} = {number(lower)}"
                else:
                    sign = "<=" if item["upper_inclusive"] else "<"
                    condition = f"{number(lower)} <= {name} {sign} {number(upper)}"
                lines.extend([
                    f"    {condition} | {item['sample_count']} | "
                    f"{number(item['y_mean'])} | {number(item['y_std'])} | "
                    f"{interval(item['y_range'])}",
                ])
            lines.extend([
                f"  One-variable held-out experiment ({predictor}):",
                f"    R2 (mean)={number(relationship['one_variable_test_r2_mean'])}",
                f"    R2 (std)={number(relationship['one_variable_test_r2_std'])}",
                f"    Success folds={relationship['one_variable_test_success_folds']}/{settings['n_folds']}",
            ])
            if strongest := relationship.get("strongest_residual_association_after_one_variable_fit"):
                lines.extend([
                    "  Largest residual correlation among other analyzed variables "
                    "(criterion=max(|Pearson|, |Spearman|)):",
                    f"    variable={strongest['variable']}",
                    f"    Pearson={number(strongest['pearson'])}",
                    f"    Spearman={number(strongest['spearman'])}",
                ])
        if matrix := result.get("pairwise_correlations"):
            lines.append("Pairwise correlations (Pearson linear; Spearman rank; pairwise finite samples):")
            for left in range(len(matrix["variables"])):
                for right in range(left + 1, len(matrix["variables"])):
                    lines.append(
                        f"  {matrix['variables'][left]} ↔ {matrix['variables'][right]}: "
                        f"Pearson={number(matrix['pearson'][left][right])}, "
                        f"Spearman={number(matrix['spearman'][left][right])}."
                    )
        if result["exceptions"]:
            lines.append("Exceptions:\n" + "\n".join(result["exceptions"]))
        return "\n".join(lines)
