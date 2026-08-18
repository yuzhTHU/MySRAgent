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
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate formula fit quality to data.

        This tool can return candidate formulas for submission when `y` is the target variable and `f` does not depend on the target variable.

        Args:
            f: Formula string, e.g., "x1**2 + sin(x2) + 3.5 * tanh(x3)".
                Common operators like sin, sinh, sec, sech, and sigmoid are all supported; do not use `numpy` or `np`.
            y: Target variable name. Use target variable by default.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            fit: Whether to optimize formula parameters using BFGS algorithm.
            show_diagnostics: Whether the result should include a compact residual error profile,
                the worst samples, and the strongest residual-variable correlations.
        """
        data = self.context['data']
        y = y or self.context['target']
        y = y.strip().strip('"').strip("'")
        eq_y = self.parse_formula(y)
        eq_f = self.parse_formula(f)
        y_true = np.asarray(eq_y.eval(data)).flatten()

        variables = [var for var in eq_f.iter_preorder() if isinstance(var, nd.Variable)]
        for var in variables:
            if var.name not in data:
                eq_f = eq_f.replace(var, nd.Number(np.random.rand()))
                fit = True # If there are unknown variables, we must fit the formula to data.
        
        if fit:
            nd.BFGSFit(eq_f).fit(data, y_true)

        evaluation = self.evaluate(
            f=eq_f,
            y=eq_y,
            y_true=y_true,
            show_diagnostics=show_diagnostics,
        )
        return {
            **evaluation,
            "parameters_optimized": fit,
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        text = cls.format_evaluation_result(result, title="Evaluated formula")
        if result.get("parameters_optimized"):
            text += "\nNumeric parameters were optimized on these same samples; the reported fit is in-sample."
        return text

@BaseTool.register('submit_formula')
class SubmitFormulaTool(EvaluateTool):
    metadata = ToolMetadata(
        name="submit_formula",
        description=(
            "Evaluate formula fit quality to data. "
            "If you are satisfied enough with a formula, use this tool to submit it."
            "You can submit any formula as many times as you want, but only the best formula will be considered. "
            "It returns candidate formulas only when `y` is the default target and the formula does not contain the target itself."
        ),
    )
