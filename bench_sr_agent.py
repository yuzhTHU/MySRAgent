"""
LLM-SRBench 评估脚本 (浓缩版)

将 benchmark 评估流程整合为单个脚本:
  1) 数据读取与预处理
  2) 调用符号回归函数 (placeholder)
  3) 评估结果
  4) 汇总多次运行的结果

Usage:
    # 测试单个问题
    python bench_sr_agent.py --exp_name test_my_algorithm --dataset lsrtransform --problem_names II.6.15b_1_0
    # 测试单个数据集
    python bench_sr_agent.py --exp_name test_my_algorithm --dataset bio_pop_growth
    # 测试一系列问题
    python bench_sr_agent.py --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6
    # 测试特定算法
    python bench_sr_agent.py linear --exp_name test_linear_fitting
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
import nd2py as nd
from pathlib import Path
from datetime import datetime
from socket import gethostname
from typing import Any, Callable, Dict, List, Optional
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error
from sr_agent.utils import setup_logging, log_exception, tag2ansi, seed_all, add_minus_flags, add_negation_flags, get_symbolic_acc
from run_sr_agent import build_argparser as build_sragent_argparser, sanitize_filename, save_args
from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult, Problem
from sr_agent._vendor.llmsr_bench.algorithms import get_update_parser, get_algorithm, list_algorithms

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
    parser = argparse.ArgumentParser(
        description="LLM-SRBench Evaluation Script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algorithm", default="my_sr_agent", choices=list_algorithms(), help="符号回归算法名称")
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--save_path", default=None, help="Path to save agent logs and artifacts. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (verbose + raise caught exceptions).")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data" / "llm-srbench-data"), help="HDF5 数据文件所在目录")
    parser.add_argument("--datasets", type=str, default=None, nargs="+", choices=list(DATASET_SPLITS.keys()), help="数据集名称, 默认评估全部数据集")
    parser.add_argument("--problem_names", type=str, default=None, nargs="+", help="仅评估指定问题（方程）ID, 默认评估全部问题")
    parser.add_argument("--skip_existing", action="store_true", default=False, help="如果结果文件已存在则跳过评估")
    parser.add_argument("--skip_successful", action="store_true", default=True, help="如果结果文件已存在且成功则跳过评估")
    parser.add_argument("--anonymize", action="store_true", help="Anonymize agent-facing variables as x1..xn and target as y.")
    # 解析 --alg 参数以获取对应的 update_parser
    args, _ = parser.parse_known_args()
    if (update_parser_fn := get_update_parser(args.algorithm)):
        parser = update_parser_fn(parser)
    add_minus_flags(parser)
    add_negation_flags(parser)
    return parser


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


def anonymize_problem(problem: Problem) -> Problem:
    """Return an agent-facing anonymized copy of a benchmark problem."""

    target = problem.symbols[0]
    features = problem.symbols[1:]
    feature_mapping = {name: f"x{i}" for i, name in enumerate(features, start=1)} | {target: "y"}
    
    anonymized_symbols = [feature_mapping[sym] for sym in problem.symbols]
    anonymized_symbol_descs = ["target variable", *[f"input variable {i}" for i in range(1, len(features) + 1)]]

    anonymized_expression = nd.parse(problem.expression.replace("^", "**").replace("np.", ""))
    for var in anonymized_expression.iter_preorder():
        if not isinstance(var, nd.Variable):
            pass
        elif var.name not in feature_mapping:
            pass # 可能是 pi, e 之类的常数
        else:
            var.name = feature_mapping[var.name]
    anonymized_expression = anonymized_expression.to_str()
    _logger.debug(
        f"[{problem.equation_idx} @ {problem.dataset_identifier}]"
        f"Anonymization enabled. "
        f"Variable mapping: {feature_mapping}\n"
        f"Original formula: {target} = {problem.expression}\n"
        f"Anonymized formula: {anonymized_symbols[0]} = {anonymized_expression}\n"
    )

    anonymized_problem = Problem(
        dataset_identifier=problem.dataset_identifier,
        equation_idx=problem.equation_idx,
        symbols=anonymized_symbols,
        symbol_descs=anonymized_symbol_descs,
        symbol_properties=problem.symbol_properties,
        expression=anonymized_expression,
        samples=problem.samples,
    )
    return anonymized_problem


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
        acc01 = np.mean(np.abs(y_true - y_pred) <= 0.1 * np.abs(y_true))
        return {
            "mse": mse, "nmse": nmse, "r2": r2, "acc01": acc01,
            "kdt": kdt, "mape": mape, "num_valid_points": len(y_true),
        }


def evaluate_problem(args, problem: Problem, sr_fn: Callable, exp_path: Path) -> Dict:
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

    # Symbolic Accurate
    try:
        data = {sym: problem.test_samples[:, i] for i, sym in enumerate(problem.symbols)}
        f_true = nd.parse(problem.expression.replace("^", "**").replace("np.", ""))
        f_pred = nd.parse(result.expression.replace("^", "**").replace("np.", ""))
        symbolic_acc = get_symbolic_acc(
            f_true,
            f_pred,
            data,
            return_details=True,
            llm_provider='openrouter',
            llm_model='deepseek/deepseek-v4-flash',
        )
        foo = lambda x: tag2ansi(('[green bold]EQUIVALENT[reset]' if x is True else '[red bold]NOT EQUIVALENT[reset]' if x is False else f'[gray bold]{x!r}[reset]'))
        _logger.note(tag2ansi(
            f"[{problem.equation_idx}] The predicted formula is judged to be {foo(symbolic_acc['equivalent'])} since {symbolic_acc.get('reason')}:\n"
            f"  f_true = [green]{f_true.to_str()}[reset]\n"
            f"  f_pred = [red]{f_pred.to_str()}[reset]"
            f"Manually modify [green bold]{exp_path}[reset] if you want to override the symbolic accuracy judgment."
        ))
    except Exception as e:
        symbolic_acc = {
            "equivalent": None,
            "reason": f"symbolic accuracy check failed: [{type(e).__name__}] {e}",
        }
        _logger.warning(f"[{problem.equation_idx}] Symbolic accuracy check failed: {log_exception(e)}")

    return {
        "equation_id": problem.equation_idx,
        "dataset_identifier": problem.dataset_identifier,
        "gt_expression": problem.expression,
        "discovered_expression": result.expression,
        "num_train": len(problem.train_samples),
        "num_test": len(problem.test_samples),
        "search_time": search_time,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
        "symbolic_acc": symbolic_acc['equivalent'],
        "symbolic_acc_detail": symbolic_acc['reason'],
    }


def log_result(result: Dict):
    lines = []
    lines.append(f'[gray]{"=" * 50}')
    lines.append(f"[blue bold]Problem {result['equation_id']} @ {result.get('dataset_identifier', 'Unknown')} evaluated.[reset]")
    lines.append(f'[gray]{"-" * 50}')
    lines.append(f"[blue]GT: [green]{result['gt_expression']}[reset]")
    lines.append(f"[blue]Discovered: [red]{result['discovered_expression']}[reset]")
    if (symbolic_acc := result.get("symbolic_acc")) is None:
        lines.append(f"[blue]Symbolic Accurate:[reset] [gray]N/A[reset]")
    elif symbolic_acc:
        lines.append(f"[blue]Symbolic Accurate:[reset] [green]Yes[reset]")
    else:
        lines.append(f"[blue]Symbolic Accurate:[reset] [red]No[reset]")
    if 'id_metrics' in result and result['id_metrics'] is not None:
        lines.append(
            f"[blue]In-Domain: R2={result['id_metrics'].get('r2', float('nan')):.6f}, "
            f"MSE={result['id_metrics'].get('mse', float('nan')):.6f}, "
            f"NMSE={result['id_metrics'].get('nmse', float('nan')):.6f}, "
            f"MAPE={result['id_metrics'].get('mape', float('nan')):.6f}, "
            f"Acc@0.1={result['id_metrics'].get('acc01', float('nan')):.6f}, "
            f"Valid Points={result['id_metrics'].get('num_valid_points', 0)}"
        )
    if 'ood_metrics' in result and result['ood_metrics'] is not None:
        lines.append(
            f"[blue]Out-of-Domain: R2={result['ood_metrics']['r2']:.6f}, "
            f"MSE={result['ood_metrics']['mse']:.6f}, "
            f"NMSE={result['ood_metrics']['nmse']:.6f}, "
            f"MAPE={result['ood_metrics']['mape']:.6f}, "
            f"Acc@0.1={result['ood_metrics']['acc01']:.6f}, "
            f"Valid Points={result['ood_metrics']['num_valid_points']}"
        )
    lines.append(f"[blue]Search time:[reset] {result['search_time']:.2f}s")
    lines.append(f'[gray]{"=" * 50}')
    return tag2ansi("\n".join(lines))


def aggregate_results(results: List[Dict]) -> Dict:
    """汇总多次运行 / 多个问题的结果"""

    def safe_mean(key, group):
        vals = [r[group][key] for r in results if r[group] is not None and not np.isnan(r[group][key])]
        return float(np.mean(vals)) if vals else float("nan")

    n = len(results)

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
        "avg_symbolic_acc": float(np.mean([r.get("symbolic_acc") is True for r in results])),
        "r2_hit_rates": r2_hit_rates,
    }
    summary["id_metrics"] = {
        "avg_mse": safe_mean("mse", "id_metrics"),
        "avg_nmse": safe_mean("nmse", "id_metrics"),
        "avg_r2": safe_mean("r2", "id_metrics"),
        "avg_acc01": safe_mean("acc01", "id_metrics"),
        "avg_kdt": safe_mean("kdt", "id_metrics"),
        "avg_mape": safe_mean("mape", "id_metrics"),
        "avg_num_valid_points": safe_mean("num_valid_points", "id_metrics"),
    }
    if any(r["ood_metrics"] is not None for r in results):
        summary["ood_metrics"] = {
            "avg_mse": safe_mean("mse", "ood_metrics"),
            "avg_nmse": safe_mean("nmse", "ood_metrics"),
            "avg_r2": safe_mean("r2", "ood_metrics"),
            "avg_acc01": safe_mean("acc01", "ood_metrics"),
            "avg_kdt": safe_mean("kdt", "ood_metrics"),
            "avg_mape": safe_mean("mape", "ood_metrics"),
            "avg_num_valid_points": safe_mean("num_valid_points", "ood_metrics"),
        }
    return summary

def conclude_results(results: List[Dict], llmsr_datasets: List[str], save_path: str):
    # 汇总
    results = [r for r in results if r.get("dataset_identifier") in llmsr_datasets]
    summary = aggregate_results(results)

    # 打印汇总
    lines = []
    lines.extend([
        f'[gray]{"=" * 50}[reset]',
        f"[red bold]Summary of {'|'.join(llmsr_datasets)} ({summary['total_problems']} problems)[reset]",
        f'[gray]{"-" * 50}[reset]',
        f"  [red]Avg R2   (In-Domain):[reset] {summary['id_metrics']['avg_r2']:.6f}",
        f"  [red]Avg MSE  (In-Domain):[reset] {summary['id_metrics']['avg_mse']:.6f}",
        f"  [red]Avg NMSE (In-Domain):[reset] {summary['id_metrics']['avg_nmse']:.6f}",
        f"  [red]Avg MAPE (In-Domain):[reset] {summary['id_metrics']['avg_mape']:.6f}",
        f"  [red]Avg KDT  (In-Domain):[reset] {summary['id_metrics']['avg_kdt']:.6f}",
        f"  [red]Avg Acc@0.1 (In-Domain):[reset] {summary['id_metrics']['avg_acc01']:.6f}",
        f"  [red]Avg Valid Points (In-Domain):[reset] {summary['id_metrics']['avg_num_valid_points']:.1f}",
    ])
    if "ood_metrics" in summary:
        lines.extend([
            f'[gray]{"-" * 50}[reset]',
            f"  [red]Avg R2   (Out-of-Domain):[reset] {summary['ood_metrics']['avg_r2']:.6f}",
            f"  [red]Avg MSE  (Out-of-Domain):[reset] {summary['ood_metrics']['avg_mse']:.6f}",
            f"  [red]Avg NMSE (Out-of-Domain):[reset] {summary['ood_metrics']['avg_nmse']:.6f}",
            f"  [red]Avg MAPE (Out-of-Domain):[reset] {summary['ood_metrics']['avg_mape']:.6f}",
            f"  [red]Avg KDT  (Out-of-Domain):[reset] {summary['ood_metrics']['avg_kdt']:.6f}",
            f"  [red]Avg Acc@0.1 (Out-of-Domain):[reset] {summary['ood_metrics']['avg_acc01']:.6f}",
            f"  [red]Avg Valid Points (Out-of-Domain):[reset] {summary['ood_metrics']['avg_num_valid_points']:.1f}",
        ])
    lines.append(f'[gray]{"-" * 50}[reset]')
    for thr, info in summary["r2_hit_rates"].items():
        lines.append(f"  [red]{thr:>10}:[reset] {info['count']:>5}/{summary['total_problems']} ({info['rate']:.1%})")
    lines.append(f'[gray]{"-" * 50}[reset]')
    lines.append(f"  [red]Avg Search Time:[reset] {summary['avg_search_time']:.2f}s")
    lines.append(f"  [red]Avg Symbolic Accurate Rate:[reset] {summary['avg_symbolic_acc']:.6f}")
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
    
    # 将 agent-facing 问题匿名化。样本矩阵列顺序不变，只替换符号名和自然语言描述。
    if args.anonymize:
        problems = [anonymize_problem(problem) for problem in problems]
        _logger.note("Anonymization enabled for benchmark tasks.")

    # 获取算法的 run 函数
    sr_fn = get_algorithm(args.algorithm)

    # 逐个问题运行 SR 并评估
    results = []
    for i, problem in enumerate(problems):
        _logger.note(f"[{i+1}/{len(problems)}] {problem.equation_idx}: {problem.expression}")
        exp_path = Path(args.save_path) / "results" / f"{problem.dataset_identifier}_{problem.equation_idx}.jsonl"
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        if exp_path.exists():
            lines = [json.loads(line) for line in exp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            successful_lines = [line for line in lines if 'error' not in line]
            if args.skip_successful and successful_lines:
                result = successful_lines[-1]
                results.append(result)
                _logger.info(tag2ansi(
                    f"Successful result already exists at {exp_path} ([red bold]{len(successful_lines)} successful records), skipping...\n"
                    f"Last successful result:\n{log_result(result)}"
                ))
                continue
            if args.skip_existing and lines:
                result = successful_lines[-1] if successful_lines else lines[-1]
                results.append(result)
                _logger.info(tag2ansi(
                    f"Result already exists at {exp_path} ([red bold]{len(lines)} records with [red bold]{len(successful_lines)} successful), skipping...\n"
                    f"Last result:\n{log_result(result)}"
                ))
                continue

        try:
            result = evaluate_problem(args, problem, sr_fn, exp_path)
            results.append(result)
            _logger.note(f"\n{log_result(result)}")
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
                    "kdt": float("nan"), "mape": float("nan"), "acc01": float("nan"),
                    "num_valid_points": 0
                },
                "ood_metrics": None,
                "symbolic_acc": False,
                "error": str(e),
            }
            results.append(result)
            with open(exp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str, allow_nan=True) + "\n")
            if args.debug: raise

    # 汇总结果
    for dataset in args.datasets:
        conclude_results(results, [dataset], save_path=Path(args.save_path) / "summary" / f"{dataset}.json")
    conclude_results(results, list(DATASET_SPLITS.keys()), save_path=Path(args.save_path) / "summary" / f"all.json")


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
