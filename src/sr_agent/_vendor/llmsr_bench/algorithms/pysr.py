"""
PySR 符号回归方法
"""
from __future__ import annotations
import os
import re
import shutil
import logging
import argparse
import tempfile
import numpy as np
from typing import Any
from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult

_logger = logging.getLogger(f"sr_agent.{__name__}")


def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """更新 parser，添加 PySR 相关参数。"""
    parser.add_argument("--pysr_timeout", type=int, default=120, help="PySR 每个问题的搜索时间上限（秒）。")
    parser.add_argument("--pysr_niterations", type=int, default=1000000, help="PySR 进化迭代次数。")
    parser.add_argument("--pysr_maxsize", type=int, default=30, help="PySR 表达式最大复杂度。")
    parser.add_argument("--pysr_populations", type=int, default=31, help="PySR 种群数量。")
    parser.add_argument("--pysr_max_samples", type=int, default=500, help="拟合时最多使用的训练样本数；<=0 表示使用全部样本。")
    parser.add_argument("--pysr_binary_operators", nargs="+", default=["+", "-", "*", "/"], help='PySR 二元算子列表')
    parser.add_argument("--pysr_unary_operators", nargs="+", default=["sin", "cos", "exp", "log", "sqrt", "square"], help="PySR 一元算子列表")
    parser.add_argument("--pysr_model_selection", default="best", choices=["best", "accuracy", "score"], help="PySR 模型选择策略。")
    return parser


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    """使用 PySRRegressor 执行符号回归。"""
    try:
        os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
        from pysr import PySRRegressor
    except ImportError as exc:
        raise ImportError(
            "PySR is not installed in the current environment. "
            'Install optional tool dependencies with: pip install -e ".[tools]"'
        ) from exc

    features = task.symbols[1:]
    X_fit, y_fit = task.train_X, task.train_y

    max_samples = getattr(args, "pysr_max_samples", 500)
    if max_samples is not None and max_samples > 0 and len(y_fit) > max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_fit), size=max_samples, replace=False)
        X_fit = X_fit[idx]
        y_fit = y_fit[idx]

    internal_names = [f"x{i + 1}" for i in range(X_fit.shape[1])]
    tmpdir = tempfile.mkdtemp(prefix=f"pysr_bench_{task.name}_")

    try:
        model = PySRRegressor(
            niterations=args.pysr_niterations,
            timeout_in_seconds=args.pysr_timeout,
            maxsize=args.pysr_maxsize,
            populations=args.pysr_populations,
            binary_operators=args.pysr_binary_operators,
            unary_operators=args.pysr_unary_operators,
            model_selection=args.pysr_model_selection,
            temp_equation_file=True,
            tempdir=tmpdir,
            verbosity=1 if getattr(args, "verbose", False) else 0,
            progress=bool(getattr(args, "verbose", False)),
            parallelism="serial",
            random_state=args.seed,
        )
        model.fit(X_fit, y_fit, variable_names=internal_names)
        expression = best_expression(model, internal_names, features)
        _logger.note(f"PySR best expression for {task.name}: {expression}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    def predict(X: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    return SRResult(predict=predict, expression=expression)


def best_expression(model: Any, internal_names: list[str], original_names: list[str]) -> str:
    expression = model.get_best().equation
    expression = expression or "0"
    expression = expression.strip()
    expression = expression.replace("^", "**")
    expression = expression.replace("Abs(", "abs(")
    expression = expression.replace("square(", "pow2(")

    replacements = dict(zip(internal_names, original_names))
    placeholders = {name: f"__pysr_feature_{idx}__" for idx, name in enumerate(internal_names)}
    for name in sorted(internal_names, key=len, reverse=True):
        expression = re.sub(rf"\b{re.escape(name)}\b", placeholders[name], expression)
    for name in internal_names:
        expression = expression.replace(placeholders[name], replacements[name])
    return expression
