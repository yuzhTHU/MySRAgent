# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Evaluate and update a coding-agent baseline result JSON."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", "/tmp/sr_agent_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/sr_agent_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr_agent_mplconfig")
for cache_dir in ("HF_HOME", "HF_DATASETS_CACHE", "MPLCONFIGDIR"):
    Path(os.environ[cache_dir]).mkdir(parents=True, exist_ok=True)

import json
import math
import stat
import shutil
import argparse
import nd2py as nd
import numpy as np
from typing import Any
from pathlib import Path
from scipy.stats import kendalltau
from datetime import datetime, timezone
from bench_sr_agent import load_problems
from sr_agent.utils import get_symbolic_acc
from sklearn.metrics import mean_absolute_percentage_error


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_pred)
    y_pred, y_true = y_pred[mask], y_true[mask]
    if len(y_true) == 0:
        return {
            "mse": float("nan"),
            "nmse": float("nan"),
            "r2": float("nan"),
            "kdt": float("nan"),
            "mape": float("nan"),
            "acc01": float("nan"),
            "num_valid_points": 0,
        }

    var = np.var(y_true)
    mse = float(np.mean((y_true - y_pred) ** 2))
    nmse = float(mse / var) if var > 0 else float("nan")
    return {
        "mse": mse,
        "nmse": nmse,
        "r2": float(1 - nmse),
        "kdt": float(kendalltau(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "acc01": float(np.mean(np.abs(y_true - y_pred) <= 0.1 * np.abs(y_true))),
        "num_valid_points": int(len(y_true)),
    }


def load_public_problem(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}


def load_benchmark_problem(dataset: str, equation_id: str, data_root: str):
    for problem in load_problems(dataset, data_root):
        if problem.equation_idx == equation_id:
            return problem
    raise ValueError(f"Cannot find problem {equation_id!r} in dataset {dataset!r}.")


def expression_variables(expr: nd.Symbol) -> set[str]:
    return {node.name for node in expr.iter_preorder() if isinstance(node, nd.Variable)}


def parse_expression(expression: str) -> nd.Symbol:
    return nd.parse(expression.replace("^", "**").replace("np.", "").replace("math.", ""))


def predict_expression(expression: str, X: np.ndarray, symbols: list[str]) -> np.ndarray:
    target = symbols[0]
    features = symbols[1:]
    formula = parse_expression(expression)
    variables = expression_variables(formula)
    if target in variables:
        raise ValueError(f"Discovered expression must not reference target variable {target!r}.")
    data = {feature: X[:, i] for i, feature in enumerate(features)}
    data[target] = np.zeros(len(X))
    return formula.eval(data).flatten()


def evaluate_result(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expression = args.formula or result.get("discovered_expression") or result.get("formula")
    if not expression:
        raise ValueError("No formula found. Provide --formula or set discovered_expression in the result JSON.")

    problem_path = args.problem or result.get("problem_path")
    if problem_path:
        public_problem = load_public_problem(problem_path)
        dataset = args.dataset or result.get("dataset_identifier") or str(public_problem["dataset_identifier"])
        equation_id = args.problem_name or result.get("equation_id") or str(public_problem["equation_id"])
    else:
        public_problem = None
        dataset = args.dataset or result.get("dataset_identifier")
        equation_id = args.problem_name or result.get("equation_id")
    if not dataset or not equation_id:
        raise ValueError("dataset_identifier and equation_id are required.")

    benchmark_problem = load_benchmark_problem(dataset, equation_id, args.data_root)
    symbols = list(benchmark_problem.symbols)
    target = symbols[0]
    search_time = args.search_time
    if search_time is None:
        search_time = result.get("duration_seconds")

    y_id = benchmark_problem.test_samples[:, 0]
    X_id = benchmark_problem.test_samples[:, 1:]
    y_pred_id = predict_expression(expression, X_id, symbols)
    id_metrics = compute_metrics(y_pred_id, y_id)

    ood_metrics = None
    if benchmark_problem.ood_test_samples is not None:
        y_ood = benchmark_problem.ood_test_samples[:, 0]
        X_ood = benchmark_problem.ood_test_samples[:, 1:]
        y_pred_ood = predict_expression(expression, X_ood, symbols)
        ood_metrics = compute_metrics(y_pred_ood, y_ood)

    try:
        f_true = nd.parse(benchmark_problem.expression.replace("^", "**").replace("np.", ""))
        f_pred = parse_expression(expression)
        if target in expression_variables(f_pred):
            symbolic_acc = {
                "equivalent": False,
                "reason": f"predicted formula references target variable {target!r}",
            }
        else:
            data = {sym: benchmark_problem.test_samples[:, i] for i, sym in enumerate(symbols)}
            symbolic_acc = get_symbolic_acc(
                f_true,
                f_pred,
                data,
                return_details=True,
                llm_judge=args.llm_judge,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
    except Exception as exc:
        symbolic_acc = {
            "equivalent": None,
            "reason": f"symbolic accuracy check failed: [{type(exc).__name__}] {exc}",
        }

    result.update(
        {
            # Fields aligned with bench_sr_agent.py:evaluate_problem.
            "equation_id": equation_id,
            "dataset_identifier": dataset,
            "gt_expression": benchmark_problem.expression,
            "discovered_expression": expression,
            "num_train": int(len(benchmark_problem.train_samples)),
            "num_test": int(len(benchmark_problem.test_samples)),
            "search_time": search_time,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
            "symbolic_acc": symbolic_acc["equivalent"],
            "symbolic_acc_detail": symbolic_acc["reason"],
            # Extra metadata for coding-agent baselines.
            "agent": args.agent or result.get("agent"),
            "agent_model": args.agent_model or result.get("agent_model"),
            "agent_provider": args.agent_provider or result.get("agent_provider"),
            "token_usage": args.token_usage,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "total_tokens": args.total_tokens,
            "tool_call_count": args.tool_call_count,
            "command_count": args.command_count,
            "problem_path": str(problem_path) if problem_path else result.get("problem_path"),
            "context_path": result.get("context_path"),
            "target": target,
            "features": symbols[1:],
            "symbols": symbols,
            "symbol_descs": list(benchmark_problem.symbol_descs),
            "symbol_properties": list(benchmark_problem.symbol_properties),
            "num_ood": int(0 if benchmark_problem.ood_test_samples is None else len(benchmark_problem.ood_test_samples)),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "evaluator": "baseline/coding_agent/evaluate_result.py",
            "llm_judge": bool(args.llm_judge),
            "llm_provider": args.llm_provider if args.llm_judge else None,
            "llm_model": args.llm_model if args.llm_judge else None,
            "status": "evaluated",
            "result_locked": True,
        }
    )
    if public_problem is not None and "context_path" in public_problem:
        result.setdefault("context_path", str(public_problem["context_path"]))
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a coding-agent symbolic-regression result and update its JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result", required=True, help="Path to result JSON to update.")
    parser.add_argument("--formula", default=None, help="Formula to evaluate. Defaults to discovered_expression in --result.")
    parser.add_argument("--problem", default=None, help="Path to exported problem.npz. Defaults to problem_path in --result.")
    parser.add_argument("--dataset", default=None, help="Dataset id override.")
    parser.add_argument("--problem-name", default=None, help="Problem/equation id override.")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[2] / "data" / "llm-srbench-data"),
        help="LLM-SRBench data root.",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Use the LLM judge in get_symbolic_acc.")
    parser.add_argument("--llm-provider", default="openrouter", help="LLM provider for --llm-judge.")
    parser.add_argument("--llm-model", default="deepseek/deepseek-v4-flash", help="LLM model for --llm-judge.")
    parser.add_argument("--agent", default=None, help="Coding agent name, e.g. codex or claude-code. Defaults to the result JSON value.")
    parser.add_argument("--agent-model", required=True, help="Coding agent model name, e.g. GPT-5.5.")
    parser.add_argument("--agent-provider", default=None, help="Optional coding agent provider.")
    parser.add_argument("--token-usage", default=None, help="Optional raw token usage string or JSON object.")
    parser.add_argument("--input-tokens", type=int, default=None, help="Optional input/prompt token count.")
    parser.add_argument("--output-tokens", type=int, default=None, help="Optional output/completion token count.")
    parser.add_argument("--total-tokens", type=int, required=True, help="Exact total token count from the agent UI/session. Use 0 only if unavailable; do not estimate.")
    parser.add_argument("--tool-call-count", type=int, default=None, help="Optional number of tool calls made during search.")
    parser.add_argument("--command-count", type=int, default=None, help="Optional number of shell commands run during search.")
    parser.add_argument("--search-time", type=float, default=None, help="Optional search duration in seconds. Defaults to duration_seconds in the result JSON.")
    parser.add_argument("--archive-codex-session", action=argparse.BooleanOptionalAction, default=True, help="Archive the latest Codex session JSONL next to the result file.")
    parser.add_argument("--codex-session-archive", default=None, help="Exact path for the archived Codex session JSONL.")
    parser.add_argument("--codex-commands-archive", default=None, help="Exact path for extracted Codex shell-command JSONL.")
    parser.add_argument("--no-lock", action="store_true", help="Do not make the result JSON read-only after evaluation.")
    return parser


def latest_codex_session(codex_home: Path | None = None) -> Path | None:
    codex_home = codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return None
    sessions = [path for path in sessions_dir.rglob("*.jsonl") if path.is_file()]
    if not sessions:
        return None
    return max(sessions, key=lambda path: path.stat().st_mtime)


def archive_codex_session(
    result_path: Path,
    transcript_path: Path | None = None,
    commands_path: Path | None = None,
) -> dict[str, str] | None:
    session_path = latest_codex_session()
    if session_path is None:
        return None

    transcript_path = transcript_path or result_path.with_suffix(".codex_session.jsonl")
    commands_path = commands_path or result_path.with_suffix(".codex_commands.jsonl")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(session_path, transcript_path)

    with session_path.open("r", encoding="utf-8") as src, commands_path.open("w", encoding="utf-8") as dst:
        for line in src:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload") or {}
            if payload.get("type") != "function_call":
                continue
            name = payload.get("name")
            if name not in {"exec_command", "write_stdin"}:
                continue
            try:
                arguments = json.loads(payload.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"raw_arguments": payload.get("arguments")}
            json.dump(
                {
                    "timestamp": event.get("timestamp"),
                    "tool": name,
                    "arguments": arguments,
                    "call_id": payload.get("call_id"),
                },
                dst,
                ensure_ascii=False,
                allow_nan=True,
            )
            dst.write("\n")

    return {
        "codex_session_source": str(session_path),
        "codex_session_archive": str(transcript_path),
        "codex_commands_archive": str(commands_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    if args.token_usage:
        try:
            args.token_usage = json.loads(args.token_usage)
        except json.JSONDecodeError:
            pass
    updated = evaluate_result(result, args)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    session_archive = (
        archive_codex_session(
            result_path,
            transcript_path=Path(args.codex_session_archive) if args.codex_session_archive else None,
            commands_path=Path(args.codex_commands_archive) if args.codex_commands_archive else None,
        )
        if args.archive_codex_session
        else None
    )
    if session_archive:
        updated |= session_archive
    result_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if not args.no_lock:
        result_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=True, default=_json_default))


if __name__ == "__main__":
    main()
