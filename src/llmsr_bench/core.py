"""
LLM-SRBench 核心数据结构
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


__all__ = ["SEDTask", "SRResult", "Problem"]

@dataclass
class SEDTask:
    """
    符号回归任务的输入 — 传递给 SR 方法的最大 input set

    注意: 原始 benchmark 的搜索器方法有以下资源限制, 公平比较时需声明:
    - LLMSR: global_max_sample_num=1000 (LLM 生成的公式骨架数量上限)
    - LaSR: max_num_samples=2000 (训练数据样本数上限, 非LLM调用次数)
    - SGA: num_iters=25 × population (进化代数 × 种群大小)
    - evaluate_timeout_seconds=30 (每个公式执行超时)
    """
    name: str
    symbols: List[str]
    symbol_descs: List[str]
    symbol_properties: List[str]
    train_X: np.ndarray        # shape: (n_train, n_input_vars)
    train_y: np.ndarray        # shape: (n_train,)
    desc: Optional[str] = None


@dataclass
class SRResult:
    """
    符号回归方法的输出 — 最小 output set

    注意: 原始 benchmark (LLMSR) 会对返回的公式做常数优化:
    - 使用 BFGS 在训练数据上优化最多 10 个参数
    - 如果你的方法未做常数优化, 结果可能较差
    """
    predict: Callable[[np.ndarray], np.ndarray]  # 输入 X(n, d) → y_pred(n,)
    expression: Optional[str] = None  # 发现的公式字符串 (可选, 用于记录)


@dataclass
class Problem:
    """一个完整的 benchmark 问题, 包含训练/测试数据"""
    dataset_identifier: str
    equation_idx: str
    symbols: List[str]
    symbol_descs: List[str]
    symbol_properties: List[str]
    expression: str  # ground truth 表达式 (如 "8*pi*Ef*epsilon*r**3/(3*sin(2*theta))")
    samples: Dict[str, np.ndarray]  # {"train": ..., "test": ..., "ood_test"?}
    desc: Optional[str] = None

    @property
    def train_samples(self) -> np.ndarray:
        return self.samples["train"]

    @property
    def test_samples(self) -> np.ndarray:
        return self.samples["test"]

    @property
    def ood_test_samples(self) -> Optional[np.ndarray]:
        return self.samples.get("ood_test", None)

    def create_task(self) -> SEDTask:
        """将 Problem 转换为 SEDTask (SR 方法的输入)"""
        data = self.train_samples
        return SEDTask(
            name=self.equation_idx,
            symbols=self.symbols,
            symbol_descs=self.symbol_descs,
            symbol_properties=self.symbol_properties,
            train_X=data[:, 1:],
            train_y=data[:, 0],
            desc=self.desc,
        )
