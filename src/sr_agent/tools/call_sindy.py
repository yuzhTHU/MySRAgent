# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""SINDy 符号回归工具。

使用 PySINDy (Sparse Identification of Nonlinear Dynamics) 对数据进行符号回归。
通过将目标变量 y 视为状态变量 X 的"导数"，利用 SINDy 的稀疏回归框架
发现 y = f(X) 的简洁数学表达式。
"""
import numpy as np
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata

import logging
_logger = logging.getLogger(f'sr_agent.{__name__}')


@BaseTool.register('call_sindy')
class SINDyTool(BaseTool):
    metadata = ToolMetadata(name="call_sindy")

    def execute(
        self,
        x: List[str] = None,
        y: str = None,
        poly_degree: int = 3,
        include_trig: bool = False,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Run SINDy (Sparse Identification of Nonlinear Dynamics) to discover symbolic expressions from data.
        SINDy builds a library of candidate nonlinear functions and uses sparse regression (STLSQ) to find
        a parsimonious combination that explains the target variable.
        Best suited for polynomial, interaction, and trigonometric relationships.

        Args:
            x: List of input feature names to use. If not specified, all features except target are used.
            y: Target variable name. If not specified, the default target variable is used.
            poly_degree: Maximum polynomial degree for the feature library (1-5). Higher values find more complex relationships but are slower.
            include_trig: Whether to include sin/cos terms in the feature library. Enable this if you suspect trigonometric relationships.
            threshold: Sparsity threshold for STLSQ optimizer (0.01-1.0). Larger values produce sparser (simpler) formulas.
        """
        import pysindy as ps

        data = self.context["data"]
        y_name = y or self.context["target"]
        x_names = x or [var for var in data if var != y_name]
        exceptions = []

        X_matrix = np.column_stack([data[name] for name in x_names])
        y_vec = data[y_name].flatten()

        # Build feature library
        feature_libs = [ps.PolynomialLibrary(degree=poly_degree, include_interaction=True)]
        if include_trig:
            feature_libs.append(ps.FourierLibrary(n_frequencies=2))

        if len(feature_libs) > 1:
            lib = ps.ConcatLibrary(feature_libs)
        else:
            lib = feature_libs[0]

        try:
            model = ps.SINDy(
                feature_library=lib,
                optimizer=ps.STLSQ(threshold=threshold, alpha=0.05),
            )
            # Use SINDy's fit with x_dot=y to perform static regression y = f(X)
            model.fit(X_matrix, t=1, x_dot=y_vec.reshape(-1, 1), feature_names=x_names)

            equations = model.equations()
            if equations and len(equations) > 0:
                formula_str = self._clean_formula(equations[0], x_names)
            else:
                formula_str = "0"
                exceptions.append("SINDy returned empty equations")
        except Exception as e:
            formula_str = "0"
            exceptions.append(f"SINDy fitting failed: {type(e).__name__}: {e}")

        if formula_str and formula_str != "0":
            try:
                metrics = self.evaluate(eq=formula_str)
                is_candidate = (y_name == self.context['target']) and (y_name not in (x or []))
            except Exception as e:
                metrics = {"mse": float("inf")}
                is_candidate = False
                exceptions.append(f"Formula evaluation failed: {e}")
        else:
            metrics = {"mse": float("inf")}
            is_candidate = False

        return {
            "formula": formula_str,
            "metrics": metrics,
            "is_candidate": is_candidate,
            "method": "SINDy (STLSQ sparse regression)",
            "config": {
                "poly_degree": poly_degree,
                "include_trig": include_trig,
                "threshold": threshold,
            },
            "exceptions": exceptions,
        }

    def _clean_formula(self, equation_str: str, x_names: List[str]) -> str:
        """Clean up SINDy output formula to be compatible with nd2py."""
        import re

        formula = equation_str.strip()
        # Remove leading/trailing whitespace around operators
        formula = re.sub(r'\s+', ' ', formula)

        # SINDy outputs like " 1.000 x1^2 +  3.000 x1 x2"
        # Parse terms: coefficient * feature_product
        terms = []
        # Split on + and - while keeping the sign
        parts = re.split(r'(?=[+-])', formula)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Match coefficient and variables
            match = re.match(r'^([+-]?\s*[\d.]+)\s+(.+)$', part)
            if match:
                coef_str = match.group(1).replace(' ', '')
                coef = float(coef_str)
                var_part = match.group(2).strip()
                if abs(coef) < 1e-10:
                    continue
                # Clean variable part: "x1^2" -> "x1**2", "x1 x2" -> "x1*x2"
                var_part = var_part.replace('^', '**')
                var_part = re.sub(r'(\w)\s+(\w)', r'\1*\2', var_part)
                # Round near-integer coefficients
                if abs(coef - round(coef)) < 0.005 and abs(coef) < 1000:
                    coef = int(round(coef))
                if coef == 1:
                    terms.append(var_part)
                elif coef == -1:
                    terms.append(f"-{var_part}")
                else:
                    terms.append(f"{coef}*{var_part}")
            else:
                # Just a constant or something we can't parse
                try:
                    val = float(part)
                    if abs(val) > 1e-10:
                        terms.append(str(val))
                except ValueError:
                    if part.strip():
                        terms.append(part.strip())

        if not terms:
            return "0"
        formula = " + ".join(terms)
        formula = formula.replace("+ -", "- ")
        return formula

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        parts = [f"SINDy result (method={result['method']}, config={result['config']}):"]
        parts.append(f"  Formula: {result['formula']}")
        if result['metrics'].get('mse') is not None:
            parts.append(f"  MSE: {result['metrics']['mse']:.6g}")
        if result['metrics'].get('r2') is not None:
            parts.append(f"  R²: {result['metrics']['r2']:.6g}")
        if result.get('is_candidate'):
            parts.append("  [This formula is a valid candidate for submission]")
        if result.get('exceptions'):
            parts.append(f"  Warnings: {'; '.join(result['exceptions'])}")
        return "\n".join(parts)
