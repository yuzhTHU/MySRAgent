# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据统计分析工具。

提供数据分布的基本统计量计算，包括最小值、最大值、均值、方差等。
"""

import numpy as np
from typing import Dict, Any, List, Optional

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('statistics_analysis')
class StatisticsTool(BaseTool):
    """计算输入数据和目标变量的描述性统计量。

    本工具用于快速了解数据的基本分布特征，帮助 LLM 形成对数据的初步认知。
    返回的统计量包括：最小值、最大值、均值、标准差、方差、中位数、样本数量。

    适用场景：
    - 数据分析任务的起点
    - 符号回归前的数据探索
    - 检测异常值或数据质量问题
    """

    metadata = ToolMetadata(
        name="statistics_analysis",
        description="计算数据的描述性统计量（最小值、最大值、均值、方差、标准差、中位数）。适用于快速了解数据分布特征。",
        category="statistics",
    )

    def execute(
        self,
        x_vars: Optional[List[str]] = None,
        y_var: str = "y",
    ) -> Dict[str, Any]:
        """执行统计分析。

        Args:
            x_vars: 要分析的输入特征名列表，如 ["x1", "x2"]。None 表示分析全部特征。
            y_var: 目标变量名，默认为 "y"。

        Returns:
            包含以下字段的字典：
            - target: 目标变量的统计量
            - features: 各输入特征的统计量列表
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
        """计算单个数组的统计量。

        Args:
            arr: 输入数组。
            name: 数组名称。

        Returns:
            统计量字典。
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
