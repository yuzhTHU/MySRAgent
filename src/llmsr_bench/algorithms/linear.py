"""
多项式拟合符号回归方法
"""
from __future__ import annotations

import argparse
import numpy as np
from numpy.linalg import lstsq
from typing import TYPE_CHECKING
from src.llmsr_bench.core import SEDTask, SRResult


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    """线性拟合符号回归方法

    Args: SEDTask 包含了 SR 方法需要的所有输入信息, 可以根据需要使用其中的任意部分:
    - task.name: str                    — 问题标识符
    - task.symbols: List[str]           — 所有符号名, 第一个为输出变量
    - task.symbol_descs: List[str]      — 符号的自然语言描述
    - task.symbol_properties: List[str] — 符号属性 ('O'=输出, 'V'=输入变量, 'C'=常数)
    - task.train_X: np.ndarray          — 训练输入, shape=(n_samples, n_input_vars)
    - task.train_y: np.ndarray          — 训练输出, shape=(n_samples,)
    - task.desc: Optional[str]          — 问题描述

    Returns: SRResult 包含了 SR 方法的输出:
    - predict: Callable[[np.ndarray], np.ndarray] — 输入 X, shape=(n, n_input_vars); 输出 y, shape=(n,)
    - expression: Optional[str]                   — 发现的公式字符串 (可选, 用于记录)
    """
    # 简单线性回归
    X_aug = np.column_stack([task.train_X, np.ones(len(task.train_X))])
    coeffs, _, _, _ = lstsq(X_aug, task.train_y, rcond=None)

    def predict(X: np.ndarray) -> np.ndarray:
        X_aug = np.column_stack([X, np.ones(len(X))])
        return X_aug @ coeffs

    expr_parts = " + ".join(f"{c:.4f}*{s}" for c, s in zip(coeffs[:-1], task.symbols[1:]))
    expression = f"{expr_parts} + {coeffs[-1]:.4f}"
    return SRResult(predict=predict, expression=expression)
