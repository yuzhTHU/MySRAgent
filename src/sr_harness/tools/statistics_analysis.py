# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据统计分析工具。计算变量或表达式的基本统计量，包括最小值、最大值、均值、方差等。"""
import numpy as np
import nd2py as nd
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata, is_numeric_array


@BaseTool.register('statistics_analysis')
class StatisticsTool(BaseTool):
    metadata = ToolMetadata('statistics_analysis')

    def execute(
        self,
        variables: List[str] = None,
        n_bins: int = 10,
        near_zero_threshold: float = 1e-8,
    ) -> Dict[str, Any]:
        """Execute statistical analysis.

        Args:
            variables: List of variable names to analyze, e.g., ["x1", "x2", "y"].
                Use all variables (including the target variable) by default.
                Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2", "sin(y+x1)"].
            n_bins: Number of equal-width histogram bins used to summarize each distribution (1-100).
            near_zero_threshold: First absolute-value threshold used to count near-zero samples.
                The output also reports thresholds 1e-6 and 1e-4.
        """
        data = self.context['data'] # {str: np.ndarray}, 包括 input variables & target variable
        if variables is None:
            variables = [key for key in data if is_numeric_array(data[key])]
        get_stats_args = dict(
            n_bins=max(1, min(int(n_bins), 100)),
            near_zero_threshold=max(0.0, float(near_zero_threshold)),
        )
        statistics = {}
        exceptions = []
        for item in variables:
            if item in data:
                x = data[item]
            else:
                try:
                    f = self.parse_formula(item)
                    x = f.eval(data)
                except Exception as e:
                    exceptions.append(f"Failed to compute '{item}': {str(e)}")
                    continue
            if not is_numeric_array(x):
                exceptions.append(f"Feature '{item}' did not produce numeric values.")
                continue
            try:
                statistics[item] = self.get_stats(x, **get_stats_args)
            except Exception as e:
                exceptions.append(f"Failed to analyze '{item}': {str(e)}")
        return {
            'statistics': statistics,
            'config': get_stats_args,
            'exceptions': exceptions
        }
    
    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        def number(value):
            if value == 0:
                return "0"
            return f"{value:#.3g}".removesuffix(".")

        def percent(value):
            return f"{number(100 * value)}%"

        def threshold_number(value):
            return "0" if value == 0 else f"{value:.2e}"

        def interval(left, right, last=False):
            return f"[{number(left)}, {number(right)}{']' if last else ')'}"

        sections = []
        for var, stat in result['statistics'].items():
            total = stat['n_samples']
            lines = [
                f"{var} (finite samples={stat['n_finite']}/{stat['n_samples']}; "
                f"finite ratio={percent(stat['finite_ratio'])}):",
                "  Fractions:",
                f"    Inf={stat['n_pos_inf']}/{total} ({percent(stat['n_pos_inf'] / total)})",
                f"    Negative inf={stat['n_neg_inf']}/{total} ({percent(stat['n_neg_inf'] / total)})",
                f"    NaN={stat['n_nan']}/{total} ({percent(stat['n_nan'] / total)})",
                f"    Negative={stat['n_negative']}/{total} ({percent(stat['n_negative'] / total)});",
                f"    Zero={stat['n_zero']}/{total} ({percent(stat['n_zero'] / total)});",
                f"    Positive={stat['n_positive']}/{total} ({percent(stat['n_positive'] / total)});",
                f"  Statistics (computed on {stat['n_finite']} finite values):",
                f"    Range=[{number(stat['min'])}, {number(stat['max'])}];",
                f"    Mean={number(stat['mean'])};",
                f"    Median={number(stat['median'])};",
                f"    Variance={number(stat['variance'])};",
                f"    Std={number(stat['std'])} (ddof=0);",
                f"    Q1 (25%)={number(stat['q1'])};",
                f"    Q3 (75%)={number(stat['q3'])};",
                f"  Near-zero fraction (computed on {stat['n_finite']} finite values):",
            ]
            for item in stat['near_zero_fractions']:
                lines.append(
                    f"    fraction(|{var}| <= {threshold_number(item['threshold'])})="
                    f"{item['count']}/{stat['n_finite']} ({percent(item['ratio'])});"
                )
            distribution = stat['distribution']
            bins = distribution['bins']
            collapse_note = "; near-constant values combined" if distribution.get("near_constant_collapsed") else ""
            lines.append(
                f"  Equal-width histogram (n_bins={distribution['n_bins']}; finite values only; "
                f"last bin includes its right endpoint{collapse_note}):"
            )
            lines.append("    (Range | samples | fraction of finite samples)")
            for index, item in enumerate(bins):
                ratio = item['count'] / stat['n_finite']
                lines.append(
                    f"    {interval(item['left'], item['right'], index == len(bins) - 1)} "
                    f"| {item['count']} | {percent(ratio)}"
                )
            sections.append("\n".join(lines))
        if result['exceptions']:
            sections.append("Exceptions:\n" + "\n".join(result['exceptions']))
        return "\n\n".join(sections) + ("\n" if sections else "")

    def get_stats(
        self,
        arr: np.ndarray,
        n_bins: int = 10,
        near_zero_threshold: float = 1e-8,
    ) -> Dict[str, Any]:
        """Compute statistics for a single array.

        Args:
            arr: Input array.

        Returns:
            Dictionary of statistics.
        """
        arr = np.asarray(arr).flatten()
        if arr.size == 0:
            raise ValueError("zero-size array cannot be analyzed")
        finite = np.isfinite(arr)
        values = arr[finite]
        if values.size == 0:
            raise ValueError("array contains no finite values")
        minimum, maximum = float(np.min(values)), float(np.max(values))
        near_constant = (maximum - minimum <= 32 * np.finfo(float).eps
                         * max(abs(minimum), abs(maximum), 1.0))
        if near_constant:
            counts = np.array([len(values)])
            edges = np.array([minimum, maximum])
        else:
            counts, edges = np.histogram(values, bins=n_bins)
        near_zero_thresholds = list(dict.fromkeys((float(near_zero_threshold), 1e-6, 1e-4)))
        near_zero_fractions = []
        for threshold in near_zero_thresholds:
            count = int(np.count_nonzero(np.abs(values) <= threshold))
            near_zero_fractions.append({
                "threshold": threshold,
                "count": count,
                "ratio": count / len(values),
            })
        bins = []
        for i in range(len(counts)):
            bins.append({
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "count": int(counts[i])
            })
        return {
            "n_samples": len(arr),
            "n_finite": int(values.size),
            "n_pos_inf": int(np.count_nonzero(np.isposinf(arr))),
            "n_neg_inf": int(np.count_nonzero(np.isneginf(arr))),
            "n_nan": int(np.count_nonzero(np.isnan(arr))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "negative_ratio": float(np.mean(values < 0)),
            "zero_ratio": float(np.mean(values == 0)),
            "positive_ratio": float(np.mean(values > 0)),
            "near_zero_ratio": float(np.mean(np.abs(values) <= near_zero_threshold)),
            "n_negative": int(np.count_nonzero(values < 0)),
            "n_zero": int(np.count_nonzero(values == 0)),
            "n_positive": int(np.count_nonzero(values > 0)),
            "n_near_zero": int(np.count_nonzero(np.abs(values) <= near_zero_threshold)),
            "near_zero_fractions": near_zero_fractions,
            "near_zero_threshold": float(near_zero_threshold),
            "finite_ratio": float(np.mean(finite)),
            "distribution": {"n_bins": len(bins), "bins": bins,
                             "near_constant_collapsed": bool(near_constant)},
        }
