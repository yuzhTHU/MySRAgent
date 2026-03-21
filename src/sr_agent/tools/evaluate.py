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
    """Evaluate the fit of mathematical formulas to data.

    This tool uses the nd2py symbolic engine to parse formula strings and
    evaluate their fit to given data. Returns multiple evaluation metrics
    including MSE, MAE, R², etc., to help LLM judge formula quality.

    Use cases:
    - Evaluating candidate formula quality in symbolic regression
    - Comparing fit quality of multiple formulas
    - Providing feedback to LLM for iterative formula optimization

    Supported mathematical operations:
    - Basic operations: +, -, *, /, ** (power)
    - Trigonometric functions: sin, cos, tan, asin, acos, atan
    - Hyperbolic functions: sinh, cosh, tanh
    - Exponential and logarithmic: exp, log, sqrt
    - Others: abs, sigmoid, etc.

    Note:
    - Use ** for exponentiation, not ^ (e.g., x1**2, not x1^2)
    - Variable names should match keys in input dictionary X
    """

    metadata = ToolMetadata(
        name="evaluate_formula",
        description="Evaluate how well a mathematical formula fits the data. Returns MSE, MAE, R² metrics. Formula format: 'x1**2 + sin(x2) + 3.5'. Supports basic operations, trig functions, exp/log.",
        category="evaluation",
    )

    def execute(
        self, eq: str, x_vars: Optional[List[str]] = None, y_var: str = "y", fit: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate formula fit quality.

        Args:
            eq: Formula string, e.g., "x1^2 + sin(x2) + 3.5".
            x_vars: List of input feature names, e.g., ["x1", "x2"].
                None means use all features.
            y_var: Target variable name, default is "y".
            fit: Whether to optimize formula parameters using BFGS algorithm.

        Returns:
            Dictionary containing:
            - success: Whether evaluation succeeded
            - error: Error message (if failed)
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R-squared (coefficient of determination)
            - y_pred: Predicted values array
            - formula: Simplified formula string
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
