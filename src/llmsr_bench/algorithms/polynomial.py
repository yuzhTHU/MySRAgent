"""
多项式拟合符号回归方法
"""
from __future__ import annotations

import argparse
import numpy as np
from typing import TYPE_CHECKING
from src.llmsr_bench.core import SEDTask, SRResult


def update_parser(parser):
    """更新 parser，添加多项式拟合相关参数"""
    parser.add_argument("--poly_degree", type=int, default=2, help="多项式拟合的最高阶数")
    return parser


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    """多项式拟合符号回归方法

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
    from src.llmsr_bench.core import SRResult
    from src.sr_agent.tools import PolynomialFitTool
    import nd2py as nd

    # 构建数据字典: {变量名: 数据数组}
    target = task.symbols[0]
    features = task.symbols[1:]
    data = {feat: task.train_X[:, i] for i, feat in enumerate(features)}
    data[target] = task.train_y

    # 执行多项式拟合
    tool = PolynomialFitTool(data=data, target=target)
    result = tool.execute(
        x=features,
        y=target,
        max_degree=args.poly_degree,
        include_interactions=True,
        include_bias=True,
    )

    f = nd.parse(result["formula"])

    def predict(X: np.ndarray) -> np.ndarray:
        pred_data = {feat: X[:, i] for i, feat in enumerate(features)}
        pred_data[target] = np.zeros(len(X))  # 占位，不会被使用
        return f.eval(pred_data).flatten()

    return SRResult(predict=predict, expression=result["formula"])
