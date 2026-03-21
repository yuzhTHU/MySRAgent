# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""公式评估工具。

评估数学公式对数据的拟合能力，返回多种评价指标。
"""

import numpy as np
import nd2py as nd
from typing import List, Dict, Any, Optional
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('evaluate_formula')
class EvaluateTool(BaseTool):
    """评估数学公式对数据的拟合能力。

    本工具使用 nd2py 符号引擎解析公式字符串，拟合并评估其对给定数据的拟合能力。
    返回多种评价指标，包括 MSE、MAE、R² 等，帮助 LLM 判断公式的质量。

    适用场景：
    - 符号回归中评估候选公式的质量
    - 比较多个公式的拟合效果
    - 为 LLM 提供反馈以迭代优化公式

    支持的数学运算：
    - 基本运算：+、-、*、/、**（幂）
    - 三角函数：sin、cos、tan、asin、acos、atan
    - 双曲函数：sinh、cosh、tanh
    - 指数对数：exp、log、sqrt
    - 其他：abs、sigmoid 等

    注意：
    - 幂运算使用 ** 而不是 ^（如 x1**2 而不是 x1^2）
    - 变量名应与输入字典 X 的键名一致
    """

    metadata = ToolMetadata(
        name="evaluate_formula",
        description="评估数学公式对数据的拟合能力。返回 MSE、MAE、R² 等指标。公式格式如 'x1**2 + sin(x2) + 3.5'。支持基本运算、三角函数、指数对数等。",
        category="evaluation",
    )

    def execute(
        self, eq: str, x_vars: Optional[List[str]] = None, y_var: str = "y", fit: bool = False,
    ) -> Dict[str, Any]:
        """评估公式的拟合能力。

        Args:
            eq: 公式字符串，如 "x1^2 + sin(x2) + 3.5"。
            x_vars: 输入特征名列表，如 ["x1", "x2"]。None 表示使用全部特征。
            y_var: 目标变量名，默认为 "y"。
            fit: 是否使用 BFGS 算法优化公式中的可拟合参数。

        Returns:
            包含以下字段的字典：
            - success: 是否成功评估
            - error: 错误信息（如果失败）
            - mse: 均方误差（Mean Squared Error）
            - rmse: 均方根误差（Root Mean Squared Error）
            - mae: 平均绝对误差（Mean Absolute Error）
            - r2: 决定系数（R-squared）
            - y_pred: 预测值数组
            - formula: 简化后的公式字符串
        """
        x = self.context['x']
        y = self.context['y']

        # 选择要使用的变量
        if x_vars is None:
            x_vars = list(x.keys())

        X_data = {var: x[var] for var in x_vars}
        y_data = y
        try:
            # ^ 在 Python 中是异或，** 才是幂运算，进行替换
            eq = eq.replace("^", "**")

            # 解析公式字符串为符号树
            eqtree = nd.parse(eq)

            # 如果需要拟合参数，使用 BFGS 算法优化
            if fit:
                bfgs = nd.BFGSFit(eqtree)
                bfgs.fit(X_data, y_data)
                eqtree = bfgs.expression

            # 计算预测值
            y_pred = eqtree.eval(X_data)
            y_true = np.array(y_data).flatten()
            y_pred = y_pred.flatten()

            # 计算评价指标
            mse = float(np.mean((y_pred - y_true) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(y_pred - y_true)))

            # R² = 1 - SS_res / SS_tot
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

            # 简化并输出公式
            try:
                formula_str = nd.StringPrinter(eqtree).print()
            except Exception:
                formula_str = eq

            return {
                "success": True,
                "error": None,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "y_pred": y_pred.tolist(),
                "formula": formula_str,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "mse": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "y_pred": None,
                "formula": None,
            }
