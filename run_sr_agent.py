#!/usr/bin/env python
"""Command-line entry point for running a small SRAgent experiment."""

from __future__ import annotations

import re
import os
import sys
import json
import math
import shlex
import random
import logging
import argparse
import numpy as np
import nd2py as nd
from pathlib import Path
from datetime import datetime
from socket import gethostname
from src.sr_agent import SRAgent
from src.sr_agent.tools import BaseTool
from src.sr_agent.utils import setup_logging, add_minus_flags, add_negation_flags, seed_all, log_exception, tag2ansi


SCRIPT_NAME = Path(__file__).stem  # run_sr_agent
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SRAgent on a synthetic symbolic-regression problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("-f", "--equation", default="y = sin(x1 - x2)", help="Target equation. Example: 'y = sin(x1 - x2)'.")
    parser.add_argument("--problem_description", default=None, help="Problem description passed to the agent. Defaults to one derived from --equation.")
    parser.add_argument("--features", default=None, help="Optional comma-separated feature names. Defaults to variables parsed from --equation.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--x_low", type=float, default=0.0, help="Lower bound for random features.")
    parser.add_argument("--x_high", type=float, default=1.0, help="Upper bound for random features.")
    parser.add_argument("--noise_std_ratio", type=float, default=0.0, help="Gaussian noise standard deviation added to the target.")
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name.")
    parser.add_argument("--llm_model", default="qwen/qwen3.5-flash-02-23", help="LLM model name.")
    parser.add_argument("--tools", default=BaseTool.all_registered_names, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("-K", "--local_sample_size", type=int, default=2, help="Number of LLM samples to generate for each branch.")
    parser.add_argument("-L", "--max_refinement_depth", type=int, default=5, help="Maximum agent refinement depth.")
    parser.add_argument("-C", "--global_width", type=int, default=2, help="Number of independent branches per restart loop.")
    parser.add_argument("-R", "--max_restart_loop", type=int, default=2, help="Maximum number of best-solution restart loops.")
    parser.add_argument("--restart_top_k", type=int, default=1, help="Number of previous best formulas to inject into the next restart prompt.")
    parser.add_argument("--tool_parser", default="openai", choices=["openai", "text", "json", "xml"], help="Tool response parser type.")
    parser.add_argument("--save_path", default=None, help="Path to save agent logs and artifacts. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging.")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode (verbose + raise caught exceptions).")
    parser.add_argument("--max_workers", type=int, default=0, help="Maximum number of parallel workers for tool execution. 0 means no parallel execution.")
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def sanitize_filename(value: str) -> str:
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub("_", value.strip())
    return (value or "unnamed")[:255]


def save_args(args, args_path: Path):
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists():
            i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def make_dataset(args):
    if '=' in args.equation:
        pass
    elif 'target' in args.equation:
        raise ValueError("It seems you provided an equation without '=', but it contains the word 'target'. Did you forget to format it like 'target = ...'?")
    else:
        args.equation = f'target = {args.equation}'
    target, formula_str = args.equation.split('=')
    target = target.strip()
    formula_str = formula_str.strip()
    formula = nd.parse(formula_str)
    features = set(var.name for var in formula.iter_preorder() if isinstance(var, nd.Variable))
    features = sorted(list(features))

    rng = np.random.default_rng(args.seed)
    data = {}
    for name in features:
        assert name not in data
        data[name] = rng.uniform(args.x_low, args.x_high, size=args.n_samples)
    assert target not in data
    data[target] = formula.eval(data)

    if args.noise_std_ratio > 0:
        data[target] += rng.normal(0.0, args.noise_std_ratio * np.std(data[target]), size=data[target].shape)

    return features, target, formula, data


def main(args: argparse.Namespace) -> dict:
    features, target, formula, data = make_dataset(args)
    X = {name: data[name] for name in features}
    y = {target: data[target]}
    problem_description = args.problem_description or (
        f"Find the relationship {target} = f({', '.join(features)}). "
        f"The synthetic target was generated from an unknown formula."
    )
    _logger.note(
        f"Starting experiment {args.exp_name}\n"
        f"Equation: {target} = {formula}\n"
        f"Target variable: {target}; Feature variables: {', '.join(features)}\n"
        f"Generated {args.n_samples} samples with seed {args.seed}\n"
    )

    agent = SRAgent(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        tools=args.tools,
        local_sample_size=args.local_sample_size,
        max_refinement_depth=args.max_refinement_depth,
        global_width=args.global_width,
        max_restart_loop=args.max_restart_loop,
        restart_top_k=args.restart_top_k,
        verbose=args.verbose,
        tool_parser=args.tool_parser,
        save_path=args.save_path,
        max_workers=args.max_workers,
    )

    result = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": None,
        "target_formula": f"{target} = {formula}",
        "noise_std_ratio": args.noise_std_ratio,
        "random_seed": args.seed,
        "best_formula": None,
        "best_mse": None,
        "status": "not_started",
        "progress": None,
        "token_usage": None,
        "money_usage": None,
        "tools_usage": None,
        "llm_model": f"{args.llm_model} @ {args.llm_provider}",
    }
    try:
        result |= agent.fit(X=X, y=y, problem_description=problem_description)
    except KeyboardInterrupt as e:
        _logger.note("Experiment interrupted by user.")
        result |= getattr(e, "partial_result", {"status": "interrupted"})
    except Exception as e:
        _logger.error(f"Experiment failed with an exception: {log_exception(e)}")
        result |= getattr(e, "partial_result", {"status": "failed"})
        result["error"] = repr(e)
        if args.debug: raise
    finally:
        result["duration_seconds"] = (datetime.now() - datetime.strptime(result["start_time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        result["times_usage"] = agent.named_timer.to_str(mode='time', mode_of_detail='pace', mode_of_percent='by_time')
        result["token_usage"] = agent.token_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
        result["money_usage"] = agent.money_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
        result["tools_usage"] = agent.tools_counter.to_str(mode='count', mode_of_detail='count', mode_of_percent='by_count')
        # 打印日志
        log = '\n'.join([f"[red]{k.replace("_", " ").title()}[reset]: {v}" for k, v in result.items()])
        _logger.note(tag2ansi(
            f'\n[gray]{"=" * 50}[reset]\n'
            "[red bold]Symbolic Regression Result[reset]\n"
            f"{log}\n"
            f'[gray]{"=" * 50}[reset]'
        ))
        # 保存文件
        result_path = Path(args.save_path) / "result.jsonl"
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True)
            f.write("\n")
        _logger.note(f"Result saved to {result_path}")


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
