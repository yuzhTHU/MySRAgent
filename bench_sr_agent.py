"""
LLM-SRBench 评估脚本 (浓缩版)

将 benchmark 评估流程整合为单个脚本:
  1) 数据读取与预处理
  2) 调用符号回归函数 (placeholder)
  3) 评估结果
  4) 汇总多次运行的结果

Usage:
    python bench_sr_agent.py --exp_name test_some_algorithm --dataset lsrtransform --problem_names II.6.15b_1_0
    python bench_sr_agent.py --exp_name test_some_algorithm --dataset bio_pop_growth
    python bench_sr_agent.py --exp_name test_some_algorithm
"""

from __future__ import annotations

import re
import os
import sys
import time
import json
import h5py
import shlex
import logging
import argparse
import datasets
import numpy as np
from pathlib import Path
from datetime import datetime
from socket import gethostname
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error
from src.sr_agent.utils import setup_logging, log_exception, tag2ansi, seed_all
from run_sr_agent import build_argparser as build_sragent_argparser, sanitize_filename, save_args

DATASET_SPLITS = {
    "lsrtransform": "lsr_transform",
    "bio_pop_growth": "lsr_synth_bio_pop_growth",
    "chem_react": "lsr_synth_chem_react",
    "matsci": "lsr_synth_matsci",
    "phys_osc": "lsr_synth_phys_osc",
}

SCRIPT_NAME = Path(__file__).stem  # bench_sr_agent
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.ArgumentParser:
    parser = build_sragent_argparser()
    parser.description = "LLM-SRBench Evaluation Script"
    parser.set_defaults(name=SCRIPT_NAME)
    parser.set_defaults(save_dir=f"./logs/{SCRIPT_NAME}")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data" / "llm-srbench-data"), help="HDF5 数据文件所在目录")
    parser.add_argument("--datasets", type=str, default=None, nargs="+", choices=list(DATASET_SPLITS.keys()), help="数据集名称, 默认评估全部数据集")
    parser.add_argument("--problem_names", type=str, default=None, nargs="+", help="仅评估指定问题（方程）ID, 默认评估全部问题")
    return parser


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


def load_problems(dataset_name: str, data_root: str, hf_repo_id = "nnheui/llm-srbench") -> List[Problem]:
    """
    从 HuggingFace + HDF5 加载指定数据集的所有问题

    HDF5 键名:
    - LSR-Transform: 'train', 'test'
    - LSR-Synth: 'train', 'test', 'ood_test'

    注意: 原始 benchmark 代码 (datamodules.py) 中 SynProblem 类使用了
    不同的属性名映射 (train_data, id_test_data, ood_test_data),
    但 HDF5 文件中的实际键名已统一为 train, test, ood_test。
    """
    split_name = DATASET_SPLITS[dataset_name]

    # 尝试从本地 parquet 文件加载公式元信息
    data_dir = Path(data_root) / "data"
    if not data_dir.exists():
        _logger.note(f"Local parquet not found. Downloading from {hf_repo_id}... (If you wait too long, consider downloading it manually)")
        ds = datasets.load_dataset(hf_repo_id, split=split_name, cache_dir='./data/hf_cache')
    elif not (data_path := data_dir.glob(f"{split_name}-*.parquet").__iter__().__next__()).exists():
        _logger.note(f"Local parquet not found. Downloading from {hf_repo_id}... (If you wait too long, consider downloading it manually)")
        ds = datasets.load_dataset(hf_repo_id, split=split_name, cache_dir='./data/hf_cache')
    else:
        _logger.note(f"Loading parquet from local: {data_path}")
        ds = datasets.load_dataset("parquet", data_files=str(data_path), split="train")

    # 从 HDF5 读取数值样本
    h5file_path = Path(data_root) / "lsr_bench_data.hdf5"
    problems = []
    with h5py.File(h5file_path, "r") as f:
        for entry in ds:
            name = entry["name"]
            # HDF5 路径: /lsr_transform/<name> 或 /lsr_synth/<domain>/<name>
            if split_name == "lsr_transform":
                h5_path = f"/lsr_transform/{name}"
            else:
                h5_path = f"/lsr_synth/{dataset_name}/{name}"

            samples = {k: v[...].astype(np.float64) for k, v in f[h5_path].items()}
            problems.append(Problem(
                dataset_identifier=dataset_name,
                equation_idx=name,
                symbols=entry["symbols"],
                symbol_descs=entry["symbol_descs"],
                symbol_properties=entry["symbol_properties"],
                expression=entry["expression"],
                samples=samples,
            ))
    return problems


def foo(args, task: SEDTask) -> SRResult:
    """ 符号回归方法的 placeholder。

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
    from numpy.linalg import lstsq
    X_aug = np.column_stack([task.train_X, np.ones(len(task.train_X))])
    coeffs, _, _, _ = lstsq(X_aug, task.train_y, rcond=None)

    def predict(X: np.ndarray) -> np.ndarray:
        X_aug = np.column_stack([X, np.ones(len(X))])
        return X_aug @ coeffs

    expr_parts = " + ".join(f"{c:.4f}*{s}" for c, s in zip(coeffs[:-1], task.symbols[1:]))
    expression = f"{expr_parts} + {coeffs[-1]:.4f}"
    return SRResult(predict=predict, expression=expression)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """计算 ID/OOD 测试集上的评估指标"""
    mask = ~np.isnan(y_pred) # 原始的 LLM-SRBench 就是这么做的，可能导致潜在的问题，但为了保持一致，我们也采用相同的过滤方式。
    y_pred, y_true = y_pred[mask], y_true[mask]
    if len(y_true) == 0:
        return {
            "mse": float("nan"), "nmse": float("nan"), "r2": float("nan"),
            "kdt": float("nan"), "mape": float("nan"), "num_valid_points": 0
        }
    else:
        var = np.var(y_true)
        mse = float(np.mean((y_true - y_pred) ** 2))
        nmse = float(mse / var) if var > 0 else float("nan")
        r2 = float(1 - nmse)
        kdt = float(kendalltau(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        return {
            "mse": mse, "nmse": nmse, "r2": r2,
            "kdt": kdt, "mape": mape, "num_valid_points": len(y_true),
        }


def evaluate_problem(args, problem: Problem, sr_fn: Callable[[SEDTask], SRResult]) -> Dict:
    """对单个问题运行 SR 方法并评估"""
    task = problem.create_task()

    start_time = time.time()
    result = sr_fn(args, task)
    search_time = time.time() - start_time

    # ID test
    X_id = problem.test_samples[:, 1:]
    y_id = problem.test_samples[:, 0]
    id_metrics = compute_metrics(y_pred=result.predict(X_id), y_true=y_id)

    # OOD test (if available)
    ood_metrics = None
    if problem.ood_test_samples is not None:
        X_ood = problem.ood_test_samples[:, 1:]
        y_ood = problem.ood_test_samples[:, 0]
        ood_metrics = compute_metrics(y_pred=result.predict(X_ood), y_true=y_ood)

    return {
        "equation_id": problem.equation_idx,
        "gt_expression": problem.expression,
        "discovered_expression": result.expression,
        "num_train": len(problem.train_samples),
        "num_test": len(problem.test_samples),
        "search_time": search_time,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
    }


def aggregate_results(results: List[Dict]) -> Dict:
    """汇总多次运行 / 多个问题的结果"""

    def safe_mean(key, group):
        vals = [r[group][key] for r in results if r[group] is not None and not np.isnan(r[group][key])]
        return float(np.mean(vals)) if vals else float("nan")

    if (n := len(results)) == 0:
        return {}

    # R² 达标率
    r2_thresholds = [0.5, 0.9, 0.99, 0.999]
    r2_hit_rates = {}
    for thr in r2_thresholds:
        hits = sum(1 for r in results if r["id_metrics"] is not None and r["id_metrics"]["r2"] >= thr)
        r2_hit_rates[f"r2>={thr}"] = {"count": hits, "rate": hits / n if n > 0 else 0}

    summary = {
        "dataset": results[0].get("equation_id", "").split("_")[0] if results else "",
        "total_problems": n,
        "avg_search_time": float(np.mean([r["search_time"] for r in results])),
        "r2_hit_rates": r2_hit_rates,
    }
    summary["id_metrics"] = {
        "avg_mse": safe_mean("mse", "id_metrics"),
        "avg_nmse": safe_mean("nmse", "id_metrics"),
        "avg_r2": safe_mean("r2", "id_metrics"),
        "avg_kdt": safe_mean("kdt", "id_metrics"),
        "avg_mape": safe_mean("mape", "id_metrics"),
        "avg_num_valid_points": safe_mean("num_valid_points", "id_metrics"),
    }
    if any(r["ood_metrics"] is not None for r in results):
        summary["ood_metrics"] = {
            "avg_mse": safe_mean("mse", "ood_metrics"),
            "avg_nmse": safe_mean("nmse", "ood_metrics"),
            "avg_r2": safe_mean("r2", "ood_metrics"),
            "avg_kdt": safe_mean("kdt", "ood_metrics"),
            "avg_mape": safe_mean("mape", "ood_metrics"),
            "avg_num_valid_points": safe_mean("num_valid_points", "ood_metrics"),
        }
    return summary

def conclude_results(llmsr_datasets: List[str], save_path: str):
    # 汇总
    results = []
    for dataset in llmsr_datasets:
        for exp_path in (Path(args.save_path) / "results").glob(f"{dataset}_*.jsonl"):
            with open(exp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if 'error' not in (result := json.loads(line)):
                        results.append(result)
    summary = aggregate_results(results)

    # 打印汇总
    lines = []
    lines.extend([
        f'[gray]{"=" * 50}[reset]',
        f"[red bold]Summary of {'/'.join(llmsr_datasets)} ({summary['total_problems']} problems)[reset]",
        f'[gray]{"-" * 50}[reset]',
        f"  [red]Avg search time:[reset] {summary['avg_search_time']:.2f}s",
        f"  [red]Avg R2   (In-Domain):[reset] {summary['id_metrics']['avg_r2']:.6f}",
        f"  [red]Avg NMSE (In-Domain):[reset] {summary['id_metrics']['avg_nmse']:.6f}",
        f"  [red]Avg MAPE (In-Domain):[reset] {summary['id_metrics']['avg_mape']:.6f}",
    ])
    if "ood_metrics" in summary:
        lines.extend([
            f'[gray]{"-" * 50}[reset]',
            f"  [red]Avg R2   (Out-of-Domain):[reset] {summary['ood_metrics']['avg_r2']:.6f}",
            f"  [red]Avg NMSE (Out-of-Domain):[reset] {summary['ood_metrics']['avg_nmse']:.6f}",
            f"  [red]Avg MAPE (Out-of-Domain):[reset] {summary['ood_metrics']['avg_mape']:.6f}",
        ])
    lines.append(f'[gray]{"-" * 50}[reset]')
    for thr, info in summary["r2_hit_rates"].items():
        lines.append(f"  [red]{thr:>10}:[reset] {info['count']:>5}/{summary['total_problems']} ({info['rate']:.1%})")
    lines.append(f'[gray]{"=" * 50}[reset]')
    _logger.note(tag2ansi("\n" + "\n".join(lines)))

    # 保存汇总
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, allow_nan=True)
    _logger.note(f"Summary saved to {save_path}")


def main(args: argparse.Namespace) -> dict:
    # 加载问题集
    args.datasets = args.datasets or list(DATASET_SPLITS.keys())
    problems = []
    for dataset in args.datasets:
        problems.extend(load_problems(dataset, args.data_root))
    _logger.note(f"Load {len(problems)} problems in total from datasets: {args.datasets}")

    if args.problem_names is not None:
        if unknown_problems := set(args.problem_names) - set(p.equation_idx for p in problems):
            _logger.warning(f"Unknown problems specified: {unknown_problems}")
        problems = [p for p in problems if p.equation_idx in args.problem_names]
        _logger.note(f"Filtered to {len(problems)} problems: {[p.equation_idx for p in problems]}")
        if not problems:
            _logger.error("No valid problems found after filtering. Check your --problem_names.")
            return

    # 逐个问题运行 SR 并评估
    for i, problem in enumerate(problems):
        _logger.note(f"[{i+1}/{len(problems)}] {problem.equation_idx}: {problem.expression}")
        exp_path = Path(args.save_path) / "results" / f"{problem.dataset_identifier}_{problem.equation_idx}.jsonl"
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        if exp_path.exists():
            _logger.note(f"Result already exists at {exp_path}, skipping...")
            continue
        try:
            result = evaluate_problem(args, problem, foo)
            _logger.note(
                f"R2={result["id_metrics"]['r2']:.6f}, "
                f"NMSE={result["id_metrics"]['nmse']:.6f}, "
                f"MAPE={result["id_metrics"]['mape']:.6f}, "
                f"Time={result['search_time']:.2f}s"
            )
            with open(exp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str, allow_nan=True) + "\n")
            _logger.note(f"Result saved to {exp_path}")
        except KeyboardInterrupt as e:
            _logger.warning("Evaluation interrupted by user.")
            break
        except Exception as e:
            _logger.error(f"  ERROR: {log_exception(e)}")
            result = {
                "equation_id": problem.equation_idx,
                "gt_expression": problem.expression,
                "discovered_expression": None,
                "num_train": len(problem.train_samples),
                "num_test": len(problem.test_samples),
                "search_time": float("nan"),
                "id_metrics": {
                    "mse": float("nan"), "nmse": float("nan"), "r2": float("nan"),
                    "kdt": float("nan"), "mape": float("nan"), "num_valid_points": 0
                },
                "ood_metrics": None,
                "error": str(e),
            }
            with open(exp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str, allow_nan=True) + "\n")

    # 汇总结果
    for dataset in args.datasets:
        conclude_results([dataset], save_path=Path(args.save_path) / "summary" / f"{dataset}.json")
    conclude_results(list(DATASET_SPLITS.keys()), save_path=Path(args.save_path) / "summary" / f"all.json")


if __name__ == "__main__":
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)
    save_path = Path(args.save_dir) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))

    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=save_path / "info.log",
        force=True,
    )

    if unknown:
        _logger.warning(f"Unknown args: {unknown}")
    _logger.note(f"Args: {args}")

    save_args(args, save_path / "args.json")

    main(args)
    _logger.note(tag2ansi(f"Experiment completed. Re-run the script with [green bold]{args.command}[reset]"))
