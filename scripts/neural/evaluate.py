# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "scripts" / "neural"))
import re
import os
import json
import shlex
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from socket import gethostname
from joblib import Parallel, delayed
from neural.metrics import evaluate_func
from bench_server.client import submit_benchmark
from neural.data import load_data, sample_indices
from algorithms import get_algorithm, list_algorithms, update_parser
from sr_agent.utils import add_minus_flags, add_negation_flags, seed_all, setup_logging, tag2ansi, df_to_3line

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=list_algorithms(), default="linear")
    parser.add_argument("--name", default=SCRIPT_NAME, help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/neural/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("--save_path", default=None, help="Path to save logs and artifacts. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--bench_server_url", default=os.environ.get("BENCH_SERVER_URL"), help="Optional benchmark server URL. If set, submit results and print the leaderboard.")
    parser.add_argument("--bench_timeout", type=float, default=10.0, help="Benchmark server request timeout in seconds.")
    parser.add_argument("--bench_leaderboard_limit", type=int, default=10, help="Number of leaderboard rows to request from the benchmark server.")

    parser.add_argument("--data_dir", type=str, default="data/neural/data", help="Directory containing the data files: data.npy, node_info.csv, time_info.csv.")
    parser.add_argument("--n_nodes", type=int, default=256, help="Number of nodes to sample for training & testing. Sampled from the 62130 available nodes.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[], help="Random seeds to sample nodes.")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds to sample if --seeds is not provided.")
    parser.add_argument("--parallel", action="store_true", help="Evaluate seeds in parallel with multiple worker processes.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes when --parallel is set. 0 means min(n_seeds, CPU count).")
    parser.add_argument("--stimulus", choices=["all", "Awake", "REM", "NREM"], default="Awake", help="Which kinds of time steps to use from time_info.csv.")
    parser.add_argument("--sampling_hz", type=float, default=4.0, help="Sampling frequency of the data in Hz, used to convert steps to seconds in evaluation results.")
    parser.add_argument('--normalize', action='store_true', default=True, help="Whether to normalize the data before training.")
    parser.add_argument("--hist_steps", type=int, default=1, help="Number of historical steps required by the algorithm predictor.")
    parser.add_argument("--max_rollout_steps", type=int, default=10, help="Maximum number of steps to rollout for evaluation.")

    args, _ = parser.parse_known_args()
    parser = update_parser(args.algorithm, parser) # 用算法特定的参数更新 parser
    parser = add_minus_flags(parser)    # 允许用 --exp-name 代替 --exp_name
    parser = add_negation_flags(parser) # 允许用 --no-debug 表示 debug=False
    return parser


def evaluate(args, data, node_info, time_info, seed) -> dict:
    seed_all(seed)
    ## 准备数据
    # 采样数据 (n_nodes 个节点, 若干时间步)
    sampled_node_idx, sampled_time_idx = sample_indices(
        node_info, time_info,
        n_nodes=args.n_nodes,
        seed=seed,
        stimulus=args.stimulus,
    )
    _logger.info(f"Sampled {len(sampled_node_idx)} nodes and {len(sampled_time_idx)} time steps for training/testing.")
    # 拆分数据 (沿时间维度分成两半, 前一半用于训练，后一半用于测试)
    time_cut = len(sampled_time_idx) // 2
    if time_cut == 0 or time_cut == len(sampled_time_idx):
        raise ValueError(f"Cannot split {len(sampled_time_idx)} time steps into two non-empty halves.")
    time_a = np.sort(sampled_time_idx[:time_cut])
    time_b = np.sort(sampled_time_idx[time_cut:])
    splits = {"train": (sampled_node_idx, time_a), "test": (sampled_node_idx, time_b)}
    _logger.info(f"Data splits created with {len(splits['train'][0])} training nodes and {len(splits['train'][1])} training time steps.")

    ## 训练模型
    # 加载算法
    algorithm = get_algorithm(args.algorithm)
    # 取出测试数据
    train_node_idx, train_time_idx = splits["train"]
    train_data = data[np.ix_(train_time_idx, train_node_idx)]
    train_node_info = node_info.iloc[train_node_idx]
    train_time_info = time_info.iloc[train_time_idx]
    # 训练模型
    _logger.info(f"Training model using algorithm '{args.algorithm}' with train_data shape {train_data.shape}...")
    model = algorithm.train_model(args, train_data, train_node_info, train_time_info)
    _logger.info(f"Model trained using algorithm '{args.algorithm}' with train_data shape {train_data.shape}.")
    # 得到预测函数
    predict_func = algorithm.get_predict_func(args, model)
    # 格式化训练结果
    if hasattr(algorithm, "format_result"):
        formated_train_result = algorithm.format_result(args, model)
    else:
        formated_train_result = "(No format_result method defined for this algorithm)"

    ## 测试模型
    # 评估性能
    _logger.info(f"Evaluating model performance on splits...")
    results_df = []
    simulations_df = []
    for split_name, (node_idx, time_idx) in splits.items():
        _logger.info(f"Evaluating split {split_name!r} with {len(node_idx)} nodes and {len(time_idx)} time steps...")
        result, simulation = evaluate_func(
            args, predict_func, 
            data[np.ix_(time_idx, node_idx)], 
            node_info.iloc[node_idx], 
            time_info.iloc[time_idx]
        )
        result_df = (
            pd.DataFrame.from_dict(result, orient="index")
            .rename_axis(index='metrics')
            .assign(split=split_name)
            .set_index("split", append=True)
            .swaplevel(axis=0)
        )
        simulation_df = simulation.assign(split=split_name)
        results_df.append(result_df)
        simulations_df.append(simulation_df)
        _logger.info(tag2ansi(
            f"[green bold]Evaluation {split_name!r} completed.[reset]\n"
            f"{df_to_3line(result_df)}\n"
        ))
    results_df = pd.concat(results_df)
    simulations_df = pd.concat(simulations_df)
    return {
        "seed": seed,
        "results_df": results_df,
        "simulations_df": simulations_df,
        "sampled_node_idx": sampled_node_idx,
        "sampled_time_idx": sampled_time_idx,
        "formated_train_result": formated_train_result,
    }


def summarize_seed_results(results_list: list[dict]) -> pd.DataFrame:
    if not results_list:
        raise ValueError("No seed results to summarize.")

    rows = []
    for results in results_list:
        seed = results["seed"]
        results_df = results["results_df"].reset_index()
        if "split" in results_df.columns:
            results_df["split"] = results_df["split"].replace({"eval": "test"})
        rows.append(results_df.assign(seed=seed))

    all_results = pd.concat(rows, ignore_index=True)
    value_cols = ["step", "seconds", "value"]
    value_cols = [col for col in value_cols if col in all_results.columns]

    def summarize_group(group: pd.DataFrame) -> dict:
        summary = {"n_seed": int(group["seed"].nunique())}
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            n = int(values.shape[0])
            mean = float(values.mean()) if n else np.nan
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            ci95 = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
            summary[col] = mean
            summary[f"95% CI of {col}"] = f"({mean - ci95:.2f}, {mean + ci95:.2f})" if n else "N/A"
        return summary

    summary_rows = []
    for (split, metric), group in all_results.groupby(["split", "metrics"], sort=False):
        summary_rows.append({
            "split": split,
            "metrics": metric,
            **summarize_group(group),
        })
    return pd.DataFrame(summary_rows).set_index(["split", "metrics"])


def main(args):
    if not args.seeds:
        args.seeds = list(range(args.n_seeds))
        _logger.info(f"No seeds provided, defaulting to seeds={args.seeds}.")

    ## 准备数据
    # 加载数据 (62130 个节点, 28810 时间步)
    _logger.info(f"Loading data from {args.data_dir}, this may take a while...")
    data, node_info, time_info = load_data(args)
    _logger.info(f"Data loaded with shape {data.shape}, node_info shape {node_info.shape}, time_info shape {time_info.shape}")

    ## 调用评估
    results_list = []
    if args.parallel:
        n_jobs = args.num_workers if args.num_workers > 0 else min(len(args.seeds), os.cpu_count() or 1)
        worker = Parallel(n_jobs=n_jobs, return_as="generator")
        tasks = [delayed(evaluate)(args, data, node_info, time_info, seed) for seed in args.seeds]
        generator = worker(tasks)
    else:
        generator = (evaluate(args, data, node_info, time_info, seed) for seed in args.seeds)
        
    for idx, results in enumerate(generator, 1):
        results_list.append(results)

        # 保存结果
        if args.save_path is not None:
            results_path = Path(args.save_path) / f"results_seed{results['seed']}.csv"
            results['results_df'].to_csv(results_path)
            _logger.note(f"Results saved to {results_path}")
            simulations_path = Path(args.save_path) / f"simulations_seed{results['seed']}.csv"
            results['simulations_df'].to_csv(simulations_path)
            _logger.note(f"Simulations saved to {simulations_path}")
        # 打印结果
        log = {
            "data_shape": data.shape,
            "node_sample_seed": results['seed'],
            "time_stimulus": args.stimulus,
            "total_n_node": len(results['sampled_node_idx']),
            "total_n_time": len(results['sampled_time_idx']),
            "algorithm": args.algorithm,
            "hist_steps": args.hist_steps,
            "max_rollout_steps": args.max_rollout_steps,
            "formated_train_result": results['formated_train_result'],
        }
        log = '\n'.join([f"[red]{k.replace("_", " ").title()}[reset]: {v}" for k, v in log.items()])
        leaderboard_text = submit_benchmark(args, results['results_df'])
        _logger.note(tag2ansi(
            f"[green bold][{idx}/{len(args.seeds)}] Evaluation completed.[reset]\n"
            f"{log}\n"
            f"{df_to_3line(results['results_df'])}\n"
            + (f"\n{leaderboard_text}" if leaderboard_text else "")
        ))

    ## 汇总结果
    summary_df = summarize_seed_results(results_list)
    if args.save_path is not None:
        summary_path = Path(args.save_path) / "summary.csv"
        summary_df.to_csv(summary_path)
        _logger.note(f"Summary saved to {summary_path}")

    formatted_results = ""
    for results in results_list:
        formatted_results += f"[green bold]Formatted Train Result(Seed {results['seed']}):[reset]\n"
        formatted_results += "\n".join("  " + line for line in results['formated_train_result'].splitlines()) + "\n\n"

    _logger.note(tag2ansi(
        f"[green bold]Average performance across {len(args.seeds)} seeds.[reset]\n"
        f"{df_to_3line(summary_df)}\n"
        f"Formatted Train Result: {formatted_results}"
    ))


def sanitize_filename(value: str) -> str:
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub("_", value.strip())
    return (value or "unnamed")[:255]


def save_args(args: argparse.Namespace, args_path: Path):
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists():
            i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)


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
    seed_all(42)
    save_path = Path(args.save_dir) / args.exp_name / args.algorithm
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
