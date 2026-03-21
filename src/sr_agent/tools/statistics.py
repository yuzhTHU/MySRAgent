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

    metadata = ToolMetadata(
        name="statistics_analysis",
        description="Compute descriptive statistics (min, max, mean, variance, std, median) for input data. Useful for quick data distribution analysis.",
        category="statistics",
    )

    def execute(
        self,
        x_vars: Optional[List[str]] = None,
        y_var: str = "y",
    ) -> Dict[str, Any]:
        """Execute statistical analysis.

        Args:
            x_vars: List of input feature names to analyze, e.g., ["x1", "x2"].
                None means analyze all features.
            y_var: Target variable name, default is "y".

        Returns:
            Dictionary containing:
            - target: Statistics for the target variable
            - features: List of statistics for each input feature
        """
        x = self.context['x']
        y = self.context['y']

        # 选择要分析的变量
        if x_vars is None:
            x_vars = list(x.keys())

        result = {
            "target": self._compute_stats(y, y_var),
            "features": [
                self._compute_stats(x[name], name) for name in x_vars
            ],
        }
        return result

    def _compute_stats(self, arr: np.ndarray, name: str) -> Dict[str, Any]:
        """Compute statistics for a single array.

        Args:
            arr: Input array.
            name: Array name.

        Returns:
            Dictionary of statistics.
        """
        arr = np.asarray(arr).flatten()
        return {
            "name": name,
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
