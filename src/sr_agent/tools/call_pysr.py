# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""PySR 符号回归工具。

调用 PySR (https://github.com/MilesCranmer/PySR) 进行基于遗传规划的符号回归，
使用 Julia 后端的 SymbolicRegression.jl 进化搜索简洁的数学表达式。
"""
import re
import numpy as np
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata

import logging
_logger = logging.getLogger(f'sr_agent.{__name__}')

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120


@BaseTool.register('call_pysr')
class PySRTool(BaseTool):
    metadata = ToolMetadata(name="call_pysr")

    def execute(
        self,
        binary_operators: List[str],
        unary_operators: List[str],
        x: List[str] = None,
        y: str = None,
        timeout: int = 30,
        maxsize: int = 25,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """Run PySR (genetic programming symbolic regression) to evolve mathematical expressions that fit the data.
        PySR uses Julia-based SymbolicRegression.jl to perform evolutionary search for symbolic formulas.
        It is powerful for discovering complex nonlinear relationships including trigonometric, exponential, sqrt, and nested functions.
        You MUST specify the binary and unary operators based on your hypothesis about the data.

        Args:
            binary_operators: List of binary operators for PySR to use. Choose from: "+", "-", "*", "/", "^". Select operators you believe are relevant to the underlying formula.
            unary_operators: List of unary operators for PySR to use. Choose from: "sin", "cos", "exp", "log", "sqrt", "square", "abs", "tanh", "sign". Select operators based on your hypothesis about the data.
            x: List of input feature names to use. If not specified, all features except target are used.
            y: Target variable name. If not specified, the default target variable is used.
            timeout: Maximum search time in seconds (default 30, max 120). Increase for harder problems.
            maxsize: Maximum expression complexity in number of nodes (10-40). Larger allows more complex formulas.
            max_samples: Maximum number of data samples to use for fitting (for speed). Data is subsampled if larger.
        """
        data = self.context["data"]
        y_name = y or self.context["target"]
        x_names = x or [var for var in data if var != y_name]
        exceptions = []

        # Clamp timeout
        timeout = max(10, min(timeout, MAX_TIMEOUT))

        X_matrix = np.column_stack([data[name] for name in x_names])
        y_vec = data[y_name].flatten()

        # Subsample if too many points
        if len(y_vec) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y_vec), size=max_samples, replace=False)
            X_fit = X_matrix[idx]
            y_fit = y_vec[idx]
        else:
            X_fit = X_matrix
            y_fit = y_vec

        formula_str = "0"
        pareto_front = []
        complexity = 0

        try:
            formula_str, pareto_front, complexity = self._run_pysr(
                X_fit, y_fit, x_names, binary_operators, unary_operators,
                timeout, maxsize
            )
        except Exception as e:
            exceptions.append(f"PySR failed: {type(e).__name__}: {e}")
            _logger.warning(f"PySR failed, trying gplearn fallback: {e}")
            try:
                formula_str = self._run_gplearn_fallback(
                    X_fit, y_fit, x_names, binary_operators, unary_operators
                )
            except Exception as e2:
                exceptions.append(f"gplearn fallback also failed: {type(e2).__name__}: {e2}")

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
            "method": "PySR (SymbolicRegression.jl)" if not any("gplearn" in e for e in exceptions) else "gplearn (fallback)",
            "complexity": complexity,
            "pareto_front": pareto_front[:5] if pareto_front else [],
            "config": {
                "timeout": timeout,
                "maxsize": maxsize,
                "binary_operators": binary_operators,
                "unary_operators": unary_operators,
            },
            "exceptions": exceptions,
        }

    def _run_pysr(self, X, y, x_names, binary_ops, unary_ops, timeout, maxsize):
        """Run PySR (Julia-based symbolic regression)."""
        import os
        os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'

        from pysr import PySRRegressor
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="pysr_")

        try:
            model = PySRRegressor(
                niterations=100,
                timeout_in_seconds=timeout,
                maxsize=maxsize,
                populations=15,
                binary_operators=binary_ops,
                unary_operators=unary_ops,
                temp_equation_file=True,
                tempdir=tmpdir,
                verbosity=0,
                progress=False,
                parallelism='serial',
                random_state=42,
            )
            model.fit(X, y, variable_names=x_names)

            # Get all equations sorted by loss, pick the one with lowest loss
            eqs_df = model.equations_
            best_row = eqs_df.loc[eqs_df['loss'].idxmin()]
            formula_str = str(best_row['equation'])
            formula_str = self._clean_pysr_formula(formula_str, x_names)
            complexity = int(best_row.get('complexity', 0))

            pareto_front = []
            for _, row in eqs_df.nsmallest(min(5, len(eqs_df)), 'loss').iterrows():
                pareto_front.append({
                    "formula": self._clean_pysr_formula(str(row['equation']), x_names),
                    "loss": float(row['loss']),
                    "complexity": int(row['complexity']),
                })

            return formula_str, pareto_front, complexity
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_gplearn_fallback(self, X, y, x_names, binary_ops, unary_ops):
        """Fallback to gplearn if PySR (Julia) is unavailable."""
        from gplearn.genetic import SymbolicRegressor

        op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
        unary_map = {"sin": "sin", "cos": "cos", "sqrt": "sqrt", "log": "log",
                     "neg": "neg", "inv": "inv"}
        func_set = []
        for op in binary_ops:
            if op in op_map:
                func_set.append(op_map[op])
        for op in unary_ops:
            if op in unary_map:
                func_set.append(unary_map[op])
        if not func_set:
            func_set = ["add", "sub", "mul", "div", "sin", "cos"]

        sr = SymbolicRegressor(
            population_size=500,
            generations=30,
            tournament_size=20,
            function_set=func_set,
            metric='mse',
            parsimony_coefficient=0.001,
            random_state=42,
            verbose=0,
            feature_names=x_names,
            stopping_criteria=1e-10,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=1.0,
            n_jobs=1,
        )
        sr.fit(X, y)
        raw_formula = str(sr._program)
        return self._clean_gplearn_formula(raw_formula)

    def _clean_pysr_formula(self, formula: str, x_names: List[str]) -> str:
        """Clean PySR output for nd2py compatibility.
        PySR already uses variable_names in output when provided via fit(),
        so we only need to handle special functions and operators.
        """
        formula = formula.strip()
        formula = re.sub(r'\bsquare\(([^)]+)\)', r'(\1)**2', formula)
        formula = formula.replace("^", "**")
        return formula if formula else "0"

    def _clean_gplearn_formula(self, formula: str) -> str:
        """Clean gplearn output for nd2py compatibility."""
        formula = formula.strip()
        formula = re.sub(r'\bneg\(([^)]+)\)', r'-(\1)', formula)
        formula = re.sub(r'\binv\(([^)]+)\)', r'1/(\1)', formula)
        return formula if formula else "0"

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        parts = [f"PySR result (method={result['method']}):"]
        parts.append(f"  Best formula: {result['formula']} (complexity={result.get('complexity', '?')})")
        if result['metrics'].get('mse') is not None:
            parts.append(f"  MSE: {result['metrics']['mse']:.6g}")
        if result['metrics'].get('r2') is not None:
            parts.append(f"  R²: {result['metrics']['r2']:.6g}")
        if result.get('is_candidate'):
            parts.append("  [This formula is a valid candidate for submission]")
        if result.get('pareto_front'):
            parts.append("  Pareto front (top candidates by accuracy):")
            for eq in result['pareto_front'][:5]:
                parts.append(f"    - {eq['formula']} (loss={eq['loss']:.6g}, complexity={eq['complexity']})")
        if result.get('exceptions'):
            parts.append(f"  Warnings: {'; '.join(result['exceptions'])}")
        return "\n".join(parts)
