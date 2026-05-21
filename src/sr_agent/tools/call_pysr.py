# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""PySR 符号回归工具。

调用 PySR (https://github.com/MilesCranmer/PySR) 进行基于遗传规划的符号回归，
使用 Julia 后端的 SymbolicRegression.jl 进化搜索简洁的数学表达式。
"""
import re
import logging
import numpy as np
import nd2py as nd
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata

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
        PySR perform evolutionary search for symbolic formulas.
        It is powerful for discovering complex nonlinear relationships including trigonometric, exponential, sqrt, and nested functions.
        However, it is also computationally intensive and requires careful tuning of operators and variables to find good formulas within reasonable time.
        You MUST specify the binary and unary operators based on your hypothesis about the data.

        Args:
            binary_operators: List of binary operators for PySR to use. Choose from: "+", "-", "*", "/", "^".
                Select operators you believe are relevant to the underlying formula.
            unary_operators: List of unary operators for PySR to use. Choose from: "sin", "cos", "exp", "log", "sqrt", "square", "abs", "tanh", "sign". Select operators based on your hypothesis about the data.
            x: List of input feature names to use. If not specified, all features except target are used.
                Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2"].
            y: Target variable name. If not specified, the default target variable is used.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            timeout: Maximum search time in seconds (default 30, max 120).
                If PySR did not find a good formula in a previous run, increase timeout (e.g., 60 or 90) to give it more search time.
            maxsize: Maximum expression complexity in number of nodes (10-40). Larger allows more complex formulas.
            max_samples: Maximum number of data samples to use for fitting (for speed). Data is subsampled if larger.
        """
        data = self.context["data"]
        y = y or self.context["target"]
        y = y.strip().strip('"').strip("'")
        x = x or [var for var in data if var != y]
        exceptions = []

        # Clamp timeout
        timeout = max(10, min(timeout, MAX_TIMEOUT))

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
        method = None

        try:
            formula_str, pareto_front, complexity = self._run_pysr(
                X_fit, y_fit, x_names, binary_operators, unary_operators,
                timeout, maxsize
            )
            method = "PySR"
        except Exception as e:
            exceptions.append(f"PySR failed: {type(e).__name__}: {e}")
            _logger.warning(f"PySR failed, trying gplearn fallback: {e}")
            try:
                formula_str = self._run_gplearn_fallback(
                    X_fit, y_fit, x_names, binary_operators, unary_operators
                )
                method = "gplearn"
            except Exception as e2:
                exceptions.append(f"gplearn fallback also failed: {type(e2).__name__}: {e2}")
                method = "failed"

        if formula_str and formula_str != "0":
            formula_str = self._restore_feature_names(formula_str, x_names, x)
            pareto_front = [
                item | {"formula": self._restore_feature_names(item["formula"], x_names, x)}
                for item in pareto_front
            ]
            metrics = self.evaluate(eq=formula_str)
            is_candidate = (y == self.context['target']) and (y not in x)
        else:
            metrics = {"mse": float("inf")}
            is_candidate = False

        # Generate retry hint if result is poor and timeout can be increased
        retry_hint = None
        mse = metrics.get("mse", float("inf"))
        if (mse > 1e-3 or formula_str == "0") and timeout < MAX_TIMEOUT:
            suggested_timeout = min(timeout * 2, MAX_TIMEOUT)
            retry_hint = (
                f"PySR did not find a good formula within {timeout}s. "
                f"Consider retrying with timeout={suggested_timeout} for more thorough search."
            )

        return {
            "formula": formula_str,
            "metrics": metrics,
            "is_candidate": is_candidate,
            "method": method,
            "complexity": complexity,
            "pareto_front": pareto_front[:5] if pareto_front else [],
            "config": {
                "timeout": timeout,
                "maxsize": maxsize,
                "binary_operators": binary_operators,
                "unary_operators": unary_operators,
            },
            "exceptions": exceptions,
            "retry_hint": retry_hint,
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

    def _restore_feature_names(
        self,
        formula: str,
        internal_names: List[str],
        original_expressions: List[str],
    ) -> str:
        """Replace PySR feature names with original variables or expressions."""
        restored = formula
        replacements = dict(zip(internal_names, original_expressions))
        placeholders = {name: f"__pysr_feature_{idx}__" for idx, name in enumerate(replacements)}
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
        if result.get('retry_hint'):
            parts.append(f"  ** Retry suggestion: {result['retry_hint']}")
        if result.get('exceptions'):
            parts.append(f"  Warnings: {'; '.join(result['exceptions'])}")
        return "\n".join(parts)
