# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""公式评估工具。

评估数学公式对数据的拟合能力，返回多种评价指标。
"""

import ast
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Optional
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('evaluate_formula')
class EvaluateTool(BaseTool):
    """Evaluate the fit of mathematical formulas to data.

    This tool uses the nd2py symbolic engine to parse formula strings and
    evaluate their fit to given data. Returns multiple evaluation metrics
    including MSE, MAE, R^2, etc., to help LLM judge formula quality.

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

    metadata = ToolMetadata(name="evaluate_formula", category="evaluation")

    def execute(
        self,
        eq: str,
        y_var: str = "y",
        fit: bool = False,
        x_vars: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate formula fit quality.

        Args:
            eq: Formula string, e.g., "x1**2 + sin(x2) + 3.5".
            y_var: Target variable name.
            fit: Whether to optimize formula parameters using BFGS algorithm.
            x_vars: Input variable subset to expose to the formula. None means all input variables.

        Returns:
            Dictionary containing:
            - metrics: Evaluation metrics (MSE, RMSE, MAE, R^2)
            - error: Error message if evaluation failed, None otherwise
            - formula: Simplified formula string (if different from input)
        """
        try:
            data = self._get_data(y_var)
            y_data = np.asarray(data[y_var]).flatten()
            x = {var: value for var, value in data.items() if var != y_var}
            if x_vars is None:
                x_vars = list(x.keys())
            X_data = {var: np.asarray(x[var]) for var in x_vars}

            # ^ 在 Python 中是异或，** 才是幂运算，进行替换
            eq = eq.replace("^", "**")
            formula, constants = self._prepare_formula(eq, fit)

            if fit and constants:
                def objective(values):
                    pred = self._eval_formula(formula, X_data, values)
                    return float(np.mean((np.asarray(pred).flatten() - y_data) ** 2))

                opt = minimize(objective, np.asarray(constants, dtype=float), method="BFGS")
                constants = opt.x.tolist()

            y_pred = self._eval_formula(formula, X_data, constants)
            y_true = y_data
            y_pred = y_pred.flatten()

            # 计算评价指标
            mse = float(np.mean((y_pred - y_true) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(y_pred - y_true)))

            # R^2 = 1 - SS_res / SS_tot
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

            # 只在公式简化后与输入不同时返回 formula
            result = {
                "metrics": {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                },
                "error": None,
            }
            if fit and constants:
                result["formula"] = self._format_fitted_formula(formula, constants)
            return result

        except Exception as e:
            return {
                "metrics": None,
                "error": str(e),
            }

    def _get_data(self, y_var: str) -> Dict[str, np.ndarray]:
        if "data" in self.context:
            return self.context["data"]
        if "x" in self.context and "y" in self.context:
            return {**self.context["x"], y_var: self.context["y"]}
        raise KeyError("data")

    def _prepare_formula(self, eq: str, fit: bool) -> tuple[str, list[float]]:
        tree = ast.parse(eq, mode="eval")
        if not fit:
            return eq, []

        constants = []

        class ReplaceConstants(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    idx = len(constants)
                    constants.append(float(node.value))
                    return ast.copy_location(ast.Name(id=f"__c{idx}", ctx=ast.Load()), node)
                return node

        tree = ReplaceConstants().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree), constants

    def _eval_formula(
        self,
        formula: str,
        data: Dict[str, np.ndarray],
        constants: list[float],
    ) -> np.ndarray:
        local_vars = dict(data)
        local_vars.update({f"__c{i}": value for i, value in enumerate(constants)})
        safe_globals = {
            "__builtins__": {},
            "abs": np.abs,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "pi": np.pi,
            "e": np.e,
            "np": np,
        }
        return np.asarray(eval(formula, safe_globals, local_vars), dtype=float)

    def _format_fitted_formula(self, formula: str, constants: list[float]) -> str:
        for idx, value in enumerate(constants):
            formula = formula.replace(f"__c{idx}", f"{value:.12g}")
        return formula

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format formula evaluation result for LLM consumption.

        Args:
            result: Tool execution result.
        """
        if result.get("error"):
            return f"Evaluation failed: {result['error']}"

        metrics = result.get("metrics") or {}
        parts = [
            "Evaluation metrics:",
            f"MSE={metrics.get('mse'):.6g}",
            f"RMSE={metrics.get('rmse'):.6g}",
            f"MAE={metrics.get('mae'):.6g}",
            f"R2={metrics.get('r2'):.6g}",
        ]
        if "formula" in result:
            parts.append(f"Formula={result['formula']}")
        return ", ".join(parts)
