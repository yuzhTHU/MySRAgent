# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据统计分析工具。计算变量或表达式的基本统计量，包括最小值、最大值、均值、方差等。"""
import numpy as np
import nd2py as nd
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('statistics_analysis')
class StatisticsTool(BaseTool):
    metadata = ToolMetadata('statistics_analysis')

    def execute(self, variables: List[str] = None) -> Dict[str, Any]:
        """Execute statistical analysis.

        Args:
            variables: List of variable names to analyze, e.g., ["x1", "x2", "y"].
                Use all variables (including the target variable) by default.
                Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2", "sin(y+x1)"].
        """
        data = self.context['data'] # {str: np.ndarray}, 包括 input variables & target variable
        if variables is None:
            variables = list(data.keys())
        statistics = {}
        exceptions = []
        for item in variables:
            if item in data:
                x = data[item]
            else:
                try:
                    f = nd.parse(item)
                    x = f.eval(data)
                except Exception as e:
                    exceptions.append(f"Failed to compute '{item}': {str(e)}")
                    continue
            statistics[item] = self.get_stats(x)
        return {
            'statistics': statistics,
            'exceptions': exceptions
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
        if result['exceptions']:
            result_str += "Exceptions:\n" + "\n".join(result['exceptions'])
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
