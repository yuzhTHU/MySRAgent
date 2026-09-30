# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""SR4MDL/MDLformer-guided symbolic-regression search tool."""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import nd2py as nd
import numpy as np

from .base_tool import BaseTool, ToolMetadata, is_numeric_array


@BaseTool.register("sr4mdl")
class SR4MDLTool(BaseTool):
    metadata = ToolMetadata(name="sr4mdl")
    REPOSITORY = "https://github.com/tsinghua-fib-lab/SR4MDL"
    BINARY_OPERATORS = {
        "+": nd.Add,
        "-": nd.Sub,
        "*": nd.Mul,
        "/": nd.Div,
    }
    UNARY_OPERATORS = {
        "sqrt": nd.Sqrt,
        "sin": nd.Sin,
        "cos": nd.Cos,
        "neg": nd.Neg,
        "inv": nd.Inv,
        "log": nd.Log,
        "exp": nd.Exp,
        "square": nd.Pow2,
        "cube": nd.Pow3,
    }

    def execute(
        self,
        binary_operators: List[str],
        unary_operators: List[str],
        x: List[str] = None,
        y: str = None,
        timeout: int = 60,
        maxsize: int = 30,
        max_samples: int = 500,
        n_iter: int = 100,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Run MDLformer-guided Monte Carlo tree search on the current data.

        Like `call_pysr`, this tool constructs a regression dataset from agent-visible
        expressions and returns an evaluated candidate formula. SR4MDL must be cloned into an
        isolated directory, `SR4MDL_HOME` must point to it, and `SR4MDL_CHECKPOINT` must point
        to its trained checkpoint (default: `$SR4MDL_HOME/weights/checkpoint.pth`).

        Args:
            binary_operators: Binary search operators chosen from "+", "-", "*", "/".
            unary_operators: Unary operators chosen from "sqrt", "sin", "cos", "neg",
                "inv", "log", "exp", "square", "cube".
            x: Input feature names or expressions. Defaults to numeric non-target columns.
            y: Target name or expression. Defaults to the configured target.
            timeout: Soft wall-time budget in seconds, checked whenever a new best tree appears.
            maxsize: Maximum expression-tree length, between 5 and 100.
            max_samples: Maximum fitting samples, between 20 and 5000.
            n_iter: Maximum MCTS iterations, between 1 and 10000.
            show_diagnostics: Whether final metrics include compact residual diagnostics.
        """
        status = self.backend_status()
        if not status["installed"]:
            raise RuntimeError(
                "SR4MDL is not configured: clone the repository and set SR4MDL_HOME."
            )
        checkpoint = Path(
            os.environ.get(
                "SR4MDL_CHECKPOINT",
                Path(status["root"]) / "weights" / "checkpoint.pth",
            )
        ).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(
                "SR4MDL checkpoint not found. Set SR4MDL_CHECKPOINT to checkpoint.pth."
            )

        unknown_binary = sorted(set(binary_operators) - self.BINARY_OPERATORS.keys())
        unknown_unary = sorted(set(unary_operators) - self.UNARY_OPERATORS.keys())
        if unknown_binary or unknown_unary:
            raise ValueError(
                f"Unsupported operators: binary={unknown_binary}, unary={unknown_unary}"
            )
        timeout = max(10, min(int(timeout), 600))
        maxsize = max(5, min(int(maxsize), 100))
        max_samples = max(20, min(int(max_samples), 5000))
        n_iter = max(1, min(int(n_iter), 10000))

        data = self.context["data"]
        y = (y or self.context["target"]).strip().strip('"').strip("'")
        x = x or [
            name for name, values in data.items()
            if name != y and is_numeric_array(values)
        ]
        target_formula = self.parse_formula(y)
        target_values = np.asarray(target_formula.eval(data)).flatten()
        if not is_numeric_array(target_values):
            raise ValueError(f"Target {y!r} did not produce numeric values.")

        internal_data = {}
        original_expressions = []
        for index, expression in enumerate(x, start=1):
            values = np.asarray(self.parse_formula(expression).eval(data)).flatten()
            if not is_numeric_array(values):
                raise ValueError(f"Feature {expression!r} did not produce numeric values.")
            if values.shape != target_values.shape:
                raise ValueError(
                    f"Feature {expression!r} shape {values.shape} does not match "
                    f"target shape {target_values.shape}."
                )
            internal_data[f"x{index}"] = values
            original_expressions.append(expression)
        if not internal_data:
            raise ValueError("No valid input variables available for fitting.")

        if len(target_values) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(target_values), size=max_samples, replace=False)
            internal_data = {
                name: values[indices] for name, values in internal_data.items()
            }
            target_values = target_values[indices]

        formula = self._run_sr4mdl(
            root=Path(status["root"]),
            checkpoint=checkpoint,
            X=internal_data,
            y=target_values,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            timeout=timeout,
            maxsize=maxsize,
            n_iter=n_iter,
        )
        formula = self._restore_feature_names(formula, original_expressions)
        evaluation = self.evaluate(
            f=self.parse_formula(formula),
            y=target_formula,
            show_diagnostics=show_diagnostics,
        )
        return {
            **evaluation,
            "method": "SR4MDL-MCTS",
            "backend": status,
            "config": {
                "timeout": timeout,
                "maxsize": maxsize,
                "max_samples": max_samples,
                "n_iter": n_iter,
                "binary_operators": binary_operators,
                "unary_operators": unary_operators,
            },
        }

    def _run_sr4mdl(
        self,
        *,
        root: Path,
        checkpoint: Path,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        binary_operators: List[str],
        unary_operators: List[str],
        timeout: int,
        maxsize: int,
        n_iter: int,
    ) -> str:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            import torch
            import nd2py.utils as nd2_utils

            # SR4MDL pins an older nd2py that re-exported get_fig from utils.
            # Keep SRAgent's newer nd2py installed and restore only that legacy export.
            if not hasattr(nd2_utils, "get_fig"):
                from nd2py.utils.plot import get_fig

                nd2_utils.get_fig = get_fig
            nd2_utils.init_logger(
                "sr4mdl",
                exp_name="SRAgent",
                info_level="warning",
            )
            from sr4mdl.env import Tokenizer
            from sr4mdl.model import MDLformer
            from sr4mdl.search import MCTS4MDL
        except ImportError as exc:
            raise RuntimeError(
                "SR4MDL dependencies are unavailable in this Python environment."
            ) from exc

        device = os.environ.get(
            "SR4MDL_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        args = SimpleNamespace(
            dropout=0.0,
            d_model=512,
            d_input=64,
            d_output=512,
            n_TE_layers=8,
            max_len=maxsize,
            max_param=5,
            max_var=10,
            uniform_sample_number=len(y),
            device=device,
            use_SENet=True,
            use_old_model=False,
        )
        tokenizer = Tokenizer(-100, 100, 4, args.max_var)
        model = MDLformer(args, state["xy_token_list"])
        model.load(state["xy_encoder"], state["xy_token_list"], strict=True)
        model.eval()
        estimator = MCTS4MDL(
            tokenizer=tokenizer,
            model=model,
            n_iter=n_iter,
            max_len=maxsize,
            sample_num=len(y),
            binary=[self.BINARY_OPERATORS[name] for name in binary_operators],
            unary=[self.UNARY_OPERATORS[name] for name in unary_operators],
            leaf=[nd.Number(1), nd.Number(2), nd.Number(np.pi)],
            keep_vars=True,
            normalize_y=False,
            normalize_all=False,
            remove_abnormal=True,
            train_eval_split=0.25,
            random_state=42,
        )
        started = time.monotonic()

        def early_stop(r2, complexity, equation):
            del complexity, equation
            return r2 > 0.99999 or time.monotonic() - started >= timeout

        estimator.fit(X, y, use_tqdm=False, early_stop=early_stop)
        if estimator.eqtree is None:
            raise RuntimeError("SR4MDL completed without producing a formula.")
        return estimator.eqtree.to_str()

    @staticmethod
    def _restore_feature_names(
        formula: str,
        original_expressions: List[str],
    ) -> str:
        for index in range(len(original_expressions), 0, -1):
            formula = re.sub(
                rf"\bx{index}\b",
                f"({original_expressions[index - 1]})",
                formula,
            )
        return formula

    @classmethod
    def backend_status(cls) -> Dict[str, Any]:
        configured = os.environ.get("SR4MDL_HOME")
        local_root = Path.cwd() / "third-party" / "SR4MDL"
        root = (
            Path(configured).expanduser().resolve()
            if configured
            else local_root.resolve() if (local_root / "regressor.py").is_file()
            else None
        )
        marker = "regressor.py"
        checkpoint = (
            Path(os.environ.get(
                "SR4MDL_CHECKPOINT",
                root / "weights" / "checkpoint.pth",
            ))
            if root else None
        )
        return {
            "backend": "sr4mdl",
            "repository": cls.REPOSITORY,
            "environment_variable": "SR4MDL_HOME",
            "root": str(root) if root else None,
            "installed": bool(root and (root / marker).is_file()),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_available": bool(checkpoint and checkpoint.is_file()),
            "marker": marker,
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        return BaseTool.format_result_dict(result)

    @classmethod
    def get_doc(cls) -> dict[str, str]:
        return {
            "name": "sr4mdl-search",
            "description": (
                "Use SR4MDL/MDLformer-guided search when description length should guide "
                "symbolic model selection."
            ),
            "content": (
                "Call `sr4mdl` with hypothesized binary/unary operators, input expressions, "
                "and a target expression, just as for `call_pysr`. SR4MDL runs an actual "
                "MDLformer-guided MCTS over the current data; it is not a data-export command. "
                "Set `SR4MDL_HOME` and `SR4MDL_CHECKPOINT` before use. Compare returned "
                "validation metrics and formula complexity with other search tools.\n\n"
                f"Source: {cls.REPOSITORY}"
            ),
        }


"""
Real invocation captured with the official checkpoint and 100 samples of ``y = a + b``:

>>> SR4MDLTool.backend_status()
{'backend': 'sr4mdl',
 'repository': 'https://github.com/tsinghua-fib-lab/SR4MDL',
 'environment_variable': 'SR4MDL_HOME', 'root': '.../third-party/SR4MDL',
 'installed': True, 'checkpoint': '.../third-party/SR4MDL/weights/checkpoint.pth',
 'checkpoint_available': True, 'marker': 'regressor.py'}

>>> result = tool.execute(
...     binary_operators=["+"], unary_operators=[], x=["a", "b"], y="y",
...     timeout=10, maxsize=10, max_samples=100, n_iter=1,
... )
>>> result["formula"], result["data_split_results"]["train"]["metrics"]
('a + b', {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'mape': 0.0,
           'r2': 1.0, 'aic': -inf, 'bic': -inf, 'pearson_r': 1.0,
           'spearman_r': 1.0, 'complexity': 3})
"""
