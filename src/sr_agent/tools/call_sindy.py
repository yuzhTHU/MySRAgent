# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""SINDy 符号回归工具。

使用 PySINDy (Sparse Identification of Nonlinear Dynamics) 对数据进行符号回归。
通过将目标变量 y 视为状态变量 X 的"导数"，利用 SINDy 的稀疏回归框架
发现 y = f(X) 的简洁数学表达式。
"""
import re
import logging
import numpy as np
import nd2py as nd
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata

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
        max_samples: int = 5000,
    ) -> Dict[str, Any]:
        """Run SINDy (Sparse Identification of Nonlinear Dynamics) to discover symbolic expressions from data.
        SINDy builds a library of candidate nonlinear functions and uses sparse regression (STLSQ) to find
        a parsimonious combination that explains the target variable.
        Best suited for polynomial, interaction, and trigonometric relationships.

        Args:
            x: List of input feature names to use. If not specified, all features except target are used.
                Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2"].
            y: Target variable name. If not specified, the default target variable is used.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            poly_degree: Maximum polynomial degree for the feature library (1-5). Higher values find more complex relationships but are slower.
            include_trig: Whether to include sin/cos terms in the feature library. Enable this if you suspect trigonometric relationships.
            threshold: Sparsity threshold for STLSQ optimizer (0.01-1.0). Larger values produce sparser (simpler) formulas.
            max_samples: Maximum number of data samples to use for fitting (for speed). Data is subsampled if larger.
        """
        data = self.context["data"]
        y = y or self.context["target"]
        y = y.strip().strip('"').strip("'")
        x = x or [var for var in data if var != y]
        exceptions = []

        poly_degree = max(1, min(poly_degree, 5))
        threshold = max(0.01, min(threshold, 1.0))

        try:
            eq_y = nd.parse(y.replace('^', '**').replace('np.', ''))
            data_y = eq_y.eval(data).flatten()
        except Exception as e:
            raise ValueError(
                f"Failed to compute target '{y}': {str(e)}" +
                "\nOther exceptions: " + "; ".join(exceptions)
            )
        y_vec = data_y

        eq_x_list = []
        data_x_list = []
        x_names = []
        for idx, xi in enumerate(x, 1):
            try:
                eq_x = nd.parse(xi.replace('^', '**').replace('np.', ''))
                data_x = eq_x.eval(data).flatten()
                eq_x_list.append(eq_x)
                data_x_list.append(data_x)
                x_names.append(f"x{idx}")
                assert data_x.shape == data_y.shape, f"Feature '{xi}' shape {data_x.shape} does not match target shape {data_y.shape}."
            except Exception as e:
                exceptions.append(f"Failed to compute feature '{xi}': {str(e)}")
        if len(eq_x_list) == 0:
            raise ValueError(
                "No valid input variables available for fitting.\n" +
                "Other exceptions: " + "; ".join(exceptions)
            )
        X_matrix = np.column_stack(data_x_list)

        if len(y_vec) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y_vec), size=max_samples, replace=False)
            X_fit = X_matrix[idx]
            y_fit = y_vec[idx]
        else:
            X_fit = X_matrix
            y_fit = y_vec

        try:
            formula_str = self._run_sindy(
                X_fit, y_fit, x_names, poly_degree, include_trig, threshold
            )
        except Exception as e:
            formula_str = "0"
            exceptions.append(f"SINDy fitting failed: {type(e).__name__}: {e}")

        if formula_str and formula_str != "0":
            formula_str = self._restore_feature_names(formula_str, x_names, x)
            metrics = self.evaluate(eq=formula_str)
            is_candidate = (y == self.context['target']) and (y not in x)
        else:
            metrics = {"mse": float("inf")}
            is_candidate = False

        return {
            "formula": formula_str,
            "metrics": metrics,
            "is_candidate": is_candidate,
            "method": "SINDy",
            "config": {
                "poly_degree": poly_degree,
                "include_trig": include_trig,
                "threshold": threshold,
                "max_samples": max_samples,
            },
            "exceptions": exceptions,
        }

    def _run_sindy(
        self,
        X,
        y,
        x_names: List[str],
        poly_degree: int,
        include_trig: bool,
        threshold: float,
    ) -> str:
        """Run PySINDy static regression and return a cleaned formula."""
        import pysindy as ps

        feature_libs = [ps.PolynomialLibrary(degree=poly_degree, include_interaction=True)]
        if include_trig:
            feature_libs.append(ps.FourierLibrary(n_frequencies=2))

        lib = ps.ConcatLibrary(feature_libs) if len(feature_libs) > 1 else feature_libs[0]
        model = ps.SINDy(
            feature_library=lib,
            optimizer=ps.STLSQ(threshold=threshold, alpha=0.05),
        )
        # Use SINDy's fit with x_dot=y to perform static regression y = f(X)
        model.fit(X, t=1, x_dot=y.reshape(-1, 1), feature_names=x_names)

        equations = model.equations()
        if not equations:
            return "0"
        return self._clean_formula(equations[0], x_names)

    def _clean_formula(self, equation_str: str, x_names: List[str]) -> str:
        """Clean up SINDy output formula to be compatible with nd2py."""
        formula = equation_str.strip()
        # Remove leading/trailing whitespace around operators
        formula = re.sub(r'\s+', ' ', formula)
        formula = re.sub(r'\+\s+-', '-', formula)

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

    def _restore_feature_names(
        self,
        formula: str,
        internal_names: List[str],
        original_expressions: List[str],
    ) -> str:
        """Replace SINDy feature names with original variables or expressions."""
        restored = formula
        replacements = dict(zip(internal_names, original_expressions))
        placeholders = {name: f"__sindy_feature_{idx}__" for idx, name in enumerate(replacements)}
        restored = self._replace_square_calls(restored)
        for name in sorted(replacements, key=len, reverse=True):
            restored = re.sub(rf"\b{re.escape(name)}\b", placeholders[name], restored)
        for name, expr in replacements.items():
            expr = expr.replace("^", "**").replace("np.", "")
            restored = restored.replace(placeholders[name], f"({expr})")
        try:
            restored = nd.parse(restored).to_str()
        except:
            _logger.warning(f"Failed to parse restored formula {restored!r}, returning unparsed version.")
        return restored

    @staticmethod
    def _replace_square_calls(formula: str) -> str:
        """Replace square(expr) with (expr)**2, preserving nested parentheses."""
        token = "square("
        while token in formula:
            start = formula.find(token)
            arg_start = start + len(token)
            depth = 1
            idx = arg_start
            while idx < len(formula) and depth:
                if formula[idx] == "(":
                    depth += 1
                elif formula[idx] == ")":
                    depth -= 1
                idx += 1
            if depth:
                break
            arg = formula[arg_start:idx - 1]
            formula = f"{formula[:start]}({arg})**2{formula[idx:]}"
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
