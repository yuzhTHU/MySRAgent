# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据统计分析工具。

提供数据分布的基本统计量计算，包括最小值、最大值、均值、方差等。
"""
import numpy as np
from typing import Dict, Any, List, Optional
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('statistics_analysis')
class StatisticsTool(BaseTool):
    """Compute descriptive statistics for input data and target variable.

    This tool is used to quickly understand the basic distribution characteristics
    of data, helping LLM form an initial understanding of the data.
    Returned statistics include: min, max, mean, std, variance, median, sample count.

    Use cases:
    - Starting point for data analysis tasks
    - Data exploration before symbolic regression
    - Detecting outliers or data quality issues
    """
    metadata = ToolMetadata('statistics_analysis')

    def execute(
        self,
        variables: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute statistical analysis.

        Args:
            variables: List of variable names to analyze, e.g., ["x1", "x2", "y"].
                None means analyze all variables (including the target variable).

        Returns:
            - statistics: statistics for each variable, including min, max, mean, variance, std, median, q1, q3, sample count.
        """
        data = self.context['data'] # {str: np.ndarray}, 包括 input variables & target variable
        if variables is None:
            variables = list(data.keys())
        return {
            'statistics': {var: self.get_stats(data[var]) for var in variables}
        }
    
    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        result_str = ''
        for var, stat in result['statistics'].items():
            result_str += (
                f"Variable '{var}': n={stat['n_samples']}, "
                f"min={stat['min']:.4f}, max={stat['max']:.4f}, "
                f"mean={stat['mean']:.4f}, variance={stat['variance']:.4f}, "
                f"std={stat['std']:.4f}, median={stat['median']:.4f}, "
                f"q1={stat['q1']:.4f}, q3={stat['q3']:.4f}\n"
            )
        return result_str

    def get_stats(self, arr: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for a single array.

        Args:
            arr: Input array.

        Returns:
            Dictionary of statistics.
        """
        arr = np.asarray(arr).flatten()
        return {
            "n_samples": len(arr),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "variance": float(np.var(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }
