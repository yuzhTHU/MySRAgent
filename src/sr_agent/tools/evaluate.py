# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""公式评估工具。

评估数学公式对数据的拟合能力，返回多种评价指标。
"""
import numpy as np
import nd2py as nd
from typing import Dict, Any
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('evaluate_formula')
class EvaluateTool(BaseTool):
    metadata = ToolMetadata(name="evaluate_formula")

    def execute(
        self,
        f: str,
        y: str = None,
        fit: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate formula fit quality to data.

        Args:
            f: Formula string, e.g., "x1**2 + sin(x2) + 3.5 * tanh(x3)".
                Common operators like sin, sinh, sec, sech, and sigmoid are all supported; do not use `numpy` or `np`.
            y: Target variable name. Use target variable by default.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            fit: Whether to optimize formula parameters using BFGS algorithm.
        """
        data = self.context['data']
        y = y or self.context['target']
        eq_y = nd.parse(y.replace("^", "**").replace('np.', ''))
        eq_f = nd.parse(f.replace("^", "**").replace('np.', ''))
        y_true = eq_y.eval(data)

        variables = [var for var in eq_f.iter_preorder() if isinstance(var, nd.Variable)]
        for var in variables:
            if var.name not in data:
                eq_f = eq_f.replace(var, nd.Number(np.random.rand()))
                fit = True # If there are unknown variables, we must fit the formula to data.
        
        if fit:
            nd.BFGSFit(eq_f).fit(data, y_true)

        y_pred = eq_f.eval(data)

        # 检查是否为有效的候选目标公式
        variables = set(var.name for var in eq_f.iter_preorder() if isinstance(var, nd.Variable))
        is_candidate = (y == self.context['target']) and (y not in variables)

        return {
            "formula": eq_f.to_str(),
            "metrics": self.evaluate(y_pred=y_pred, y_true=y_true),
            "is_candidate": is_candidate,
        }

@BaseTool.register('submit_formula')
class SubmitFormulaTool(EvaluateTool):
    metadata = ToolMetadata(
        name="submit_formula",
        description=(
            "Evaluate formula fit quality to data. "
            "If you are satisfied enough with a formula, use this tool to submit it."
            "You can submit any formula as many times as you want, but only the best formula will be considered."
        ),
    )
