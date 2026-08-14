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
            near_zero_threshold: Absolute-value threshold used to count near-zero samples.
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
            'config': {
                'n_bins': n_bins,
                'near_zero_threshold': near_zero_threshold,
            },
            'exceptions': exceptions
        }
    
    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        sections = []
        for var, stat in result['statistics'].items():
            lines = [
                f"Variable or expression: '{var}'",
                (
                    f"  Sample coverage: {stat['n_finite']} finite values out of "
                    f"{stat['n_samples']} total samples "
                    f"(finite value ratio: {stat['finite_ratio']:.1%})."
                ),
                f"  Finite-value range: minimum={stat['min']:.6g}, maximum={stat['max']:.6g}.",
                (
                    f"  Center and spread: mean={stat['mean']:.6g}, median={stat['median']:.6g}, "
                    f"variance={stat['variance']:.6g}, standard deviation={stat['std']:.6g}."
                ),
                (
                    f"  Quartiles: first quartile (25%)={stat['q1']:.6g}, "
                    f"third quartile (75%)={stat['q3']:.6g}."
                ),
                (
                    f"  Sign distribution among finite values: negative value ratio: "
                    f"{stat['negative_ratio']:.1%}; exact-zero value ratio: {stat['zero_ratio']:.1%}; "
                    f"positive value ratio: {stat['positive_ratio']:.1%}."
                ),
                (
                    f"  Near-zero values: {stat['near_zero_ratio']:.1%} of finite values have "
                    f"absolute value <= {stat['near_zero_threshold']:.6g}."
                ),
            ]
            distribution = stat['distribution']
            bins = distribution['bins']
            lines.append(
                f"  Equal-width histogram: {distribution['n_bins']} bins over finite values."
            )
            for index, item in enumerate(bins):
                closing = "]" if index == len(bins) - 1 else ")"
                ratio = item['count'] / stat['n_finite'] if stat['n_finite'] else float('nan')
                lines.append(
                    f"    Bin {index + 1}: [{item['left']:.6g}, {item['right']:.6g}{closing}; "
                    f"sample count={item['count']} ({ratio:.1%} of finite values)."
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
        counts, edges = np.histogram(values, bins=n_bins)
        bins = []
        for i in range(n_bins):
            bins.append({
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "count": int(counts[i])
            })
        return {
            "n_samples": len(arr),
            "n_finite": int(values.size),
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
            "near_zero_threshold": float(near_zero_threshold),
            "finite_ratio": float(np.mean(finite)),
            "distribution": {"n_bins": len(bins), "bins": bins},
        }
