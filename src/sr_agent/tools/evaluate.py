# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""公式评估工具。

评估数学公式对数据的拟合能力，返回多种评价指标。
"""
import nd2py as nd
from typing import Dict, Any
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('evaluate_formula')
class EvaluateTool(BaseTool):
    metadata = ToolMetadata(name="evaluate_formula")

    def execute(
        self,
        eq: str,
        y_var: str = None,
        fit: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate formula fit quality to data.

        Args:
            eq: Formula string, e.g., "x1**2 + sin(x2) + 3.5".
            y_var: Target variable name. Use target variable by default.
            fit: Whether to optimize formula parameters using BFGS algorithm.
        """
        data = self.context['data']
        y_true = data[y_var or self.context['target']]
        f = nd.parse(eq.replace("^", "**"))
        if fit:
            nd.BFGSFit(f).fit(data, y_true)
        y_pred = f.eval(data)
        return {
            "formula": f.to_str(),
            "metrics": self.evaluate(y_pred=y_pred, y_true=y_true),
        }


@BaseTool.register('submit_formula')
class SubmitFormulaTool(EvaluateTool):
    """Submit a final formula candidate."""

    metadata = ToolMetadata(
        name="submit_formula",
        description=(
            "Evaluate formula fit quality to data. "
            "If you are satisfied enough with a formula, use this tool to submit it."
        ),
    )
