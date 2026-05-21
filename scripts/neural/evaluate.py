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
from datetime import datetime
from socket import gethostname
from neural.metrics import evaluate_func
from neural.data import load_data, sample_indices, build_splits
from algorithms import get_algorithm, list_algorithms, update_parser
from bench_server.client import submit_benchmark
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode.")
    parser.add_argument("--bench_server_url", default=os.environ.get("BENCH_SERVER_URL"), help="Optional benchmark server URL. If set, submit results and print the leaderboard.")
    parser.add_argument("--bench_timeout", type=float, default=10.0, help="Benchmark server request timeout in seconds.")
    parser.add_argument("--bench_leaderboard_limit", type=int, default=10, help="Number of leaderboard rows to request from the benchmark server.")

    parser.add_argument("--data_dir", type=str, default="data/neural/data", help="Directory containing the data files: data.npy, node_info.csv, time_info.csv.")
    parser.add_argument("--n_nodes", type=int, default=256, help="Number of nodes to sample for training & testing. Sampled from the 62130 available nodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to sample nodes. Use -1 for current system time.")
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


def main(args: argparse.Namespace) -> dict:
    ## 准备数据
    # 加载数据 (62130 个节点, 28810 时间步)
    _logger.info(f"Loading data from {args.data_dir}, this may take a while...")
    data, node_info, time_info = load_data(args)
    _logger.info(f"Data loaded with shape {data.shape}, node_info shape {node_info.shape}, time_info shape {time_info.shape}")
    
    # 采样数据 (n_nodes 个节点, 若干时间步)
    sampled_node_idx, sampled_time_idx = sample_indices(
        node_info, time_info,
        n_nodes=args.n_nodes,
        seed=args.seed,
        stimulus=args.stimulus,
    )
    _logger.info(f"Sampled {len(sampled_node_idx)} nodes and {len(sampled_time_idx)} time steps for training/testing.")
    
    # 拆分数据 (沿节点/时间两个维度的中线划分为四个象限, 第一象限用于 train, 其余象限用于测试)
    splits = build_splits(sampled_node_idx, sampled_time_idx, seed=args.seed)
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
    
    # 保存结果
    if args.save_path is not None:
        results_path = Path(args.save_path) / "results.csv"
        results_df.to_csv(results_path)
        _logger.note(f"Results saved to {results_path}")
        simulations_path = Path(args.save_path) / "simulations.csv"
        simulations_df.to_csv(simulations_path)
        _logger.note(f"Simulations saved to {simulations_path}")
    
    # 打印结果
    log = {
        "data_shape": data.shape,
        "node_sample_seed": args.seed,
        "time_stimulus": args.stimulus,
        "total_n_node": len(sampled_node_idx),
        "total_n_time": len(sampled_time_idx),
        "algorithm": args.algorithm,
        "hist_steps": args.hist_steps,
        "max_rollout_steps": args.max_rollout_steps,
    }
    log = '\n'.join([f"[red]{k.replace("_", " ").title()}[reset]: {v}" for k, v in log.items()])
    leaderboard_text = submit_benchmark(args, results_df)
    _logger.note(tag2ansi(
        f"[green bold]Evaluation completed.[reset]\n"
        f"{log}\n"
        f"{df_to_3line(results_df)}\n"
        + (f"\n{leaderboard_text}" if leaderboard_text else "")
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
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)
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
