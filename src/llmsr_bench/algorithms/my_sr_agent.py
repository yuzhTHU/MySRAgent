"""
多项式拟合符号回归方法
"""
from __future__ import annotations

import json
import argparse
import numpy as np
import nd2py as nd
from pathlib import Path
from logging import getLogger
from datetime import datetime
from typing import TYPE_CHECKING
from src.sr_agent import SRAgent
from src.sr_agent.tools import BaseTool
from src.llmsr_bench.core import SEDTask, SRResult
from src.sr_agent.utils import log_exception, tag2ansi

_logger = getLogger(f'sr_agent.{__name__}')


def update_parser(parser):
    """更新 parser，添加多项式拟合相关参数"""
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name.")
    parser.add_argument("--llm_model", default="qwen/qwen3.5-flash-02-23", help="LLM model name.")
    parser.add_argument("--tools", default=BaseTool.all_registered_names, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("-K", "--local_sample_size", type=int, default=2, help="Number of LLM samples to generate for each branch.")
    parser.add_argument("-L", "--max_refinement_depth", type=int, default=10, help="Maximum agent refinement depth.")
    parser.add_argument("-C", "--global_width", type=int, default=1, help="Number of independent branches per restart loop.")
    parser.add_argument("-R", "--max_restart_loop", type=int, default=2, help="Maximum number of best-solution restart loops.")
    parser.add_argument("--restart_top_k", type=int, default=1, help="Number of previous best formulas to inject into the next restart prompt.")
    parser.add_argument("--tool_parser", default="openai", choices=["openai", "text", "json", "xml"], help="Tool response parser type.")
    parser.add_argument("--max_workers", type=int, default=0, help="Maximum number of parallel workers for tool execution. 0 means no parallel execution.")
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
    if args.save_path is None:
        exp_save_path = None
    else:
        exp_save_path = Path(args.save_path) / "experiments" / f"{task.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_save_path.mkdir(parents=True, exist_ok=True)

    # 构建数据字典: {变量名: 数据数组}
    target = task.symbols[0]
    features = task.symbols[1:]
    y = {target: task.train_y}
    X = {feat: task.train_X[:, i] for i, feat in enumerate(features)}

    # 构建问题描述
    problem_description = []
    if task.desc is not None:
        problem_description.append(f"Problem Description: {task.desc}")
    for sym, desc, prop in zip(task.symbols, task.symbol_descs, task.symbol_properties):
        problem_description.append(f"{sym} ({'Output' if prop == 'O' else 'Input Variable' if prop == 'V' else 'Constant'}): {desc}")
    problem_description = "\n".join(problem_description)
    _logger.note(f"Problem Description:\n{problem_description}")

    # 执行符号回归拟合
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
        save_path=exp_save_path,
        max_workers=args.max_workers,
    )
    result = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "problem_name": task.name,
        "duration_seconds": None,
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
        if exp_save_path is not None:
            result_path = exp_save_path / "result.jsonl"
            with open(result_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=True)
                f.write("\n")
            _logger.note(f"Result saved to {result_path}")

    f = nd.parse(result["best_formula"])

    def predict(X: np.ndarray) -> np.ndarray:
        pred_data = {feat: X[:, i] for i, feat in enumerate(features)}
        pred_data[target] = np.zeros(len(X))  # 占位，不会被使用
        return f.eval(pred_data).flatten()

    return SRResult(predict=predict, expression=result["best_formula"])
