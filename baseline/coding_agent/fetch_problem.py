# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Export one LLM-SRBench problem for a coding-agent baseline run."""
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

import re
import json
import argparse
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime
from bench_sr_agent import DATASET_SPLITS, load_problems


DEFAULT_OUTPUT_DIR = Path("/tmp/sr_agent_coding_agent")
DEFAULT_RESULT_DIR = Path("logs/baseline/coding_agent")


def sanitize_filename(value: str) -> str:
    return re.sub(r'[ <>:"/\\|?*\x00-\x1f]', "_", value.strip())[:255] or "unnamed"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_problem_description(problem) -> str:
    lines = []
    if problem.desc:
        lines.append(f"Problem Description: {problem.desc}")
    for sym, desc, prop in zip(problem.symbols, problem.symbol_descs, problem.symbol_properties):
        kind = "Output" if prop == "O" else "Input Variable" if prop == "V" else "Constant"
        lines.append(f"{sym} ({kind}): {desc}")
    return "\n".join(lines)


def select_problem(dataset: str, problem_name: str, data_root: str):
    problems = load_problems(dataset, data_root)
    for problem in problems:
        if problem.equation_idx == problem_name:
            return problem
    known = ", ".join(problem.equation_idx for problem in problems[:20])
    suffix = " ..." if len(problems) > 20 else ""
    raise ValueError(f"Unknown problem {problem_name!r} in {dataset!r}. Known examples: {known}{suffix}")


def create_initial_result(
    result_path: Path,
    problem,
    problem_path: Path,
    context_path: Path,
    tool_call_log_path: Path,
    agent: str,
    overwrite_result: bool,
) -> dict[str, Any]:
    if result_path.exists() and not overwrite_result:
        raise FileExistsError(
            f"Result file already exists: {result_path}. Use --overwrite-result to reset it."
        )
    result = {
        "agent": agent,
        "dataset_identifier": problem.dataset_identifier,
        "equation_id": problem.equation_idx,
        "problem_path": str(problem_path),
        "context_path": str(context_path),
        "tool_call_log_path": str(tool_call_log_path),
        "start_time": datetime.now().astimezone().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "discovered_expression": None,
        "status": "started",
        "notes": None,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return result


def export_problem(
    problem,
    output_dir: Path,
    result_dir: Path,
    agent: str,
    result_path: Path | None = None,
    tool_call_log_path: Path | None = None,
    overwrite_problem: bool = False,
    overwrite_result: bool = False,
) -> dict[str, Any]:
    target = problem.symbols[0]
    features = problem.symbols[1:]
    train = problem.train_samples
    test = problem.test_samples
    ood = problem.ood_test_samples

    run_dir = output_dir / sanitize_filename(f"{problem.dataset_identifier}_{problem.equation_idx}")
    run_dir.mkdir(parents=True, exist_ok=True)

    context_path = run_dir / "context.npz"
    problem_path = run_dir / "problem.npz"
    manifest_path = run_dir / "manifest.json"
    if result_path is None:
        result_path = result_dir / (
            f"{sanitize_filename(problem.dataset_identifier)}_"
            f"{sanitize_filename(problem.equation_idx)}_"
            f"{sanitize_filename(agent)}.json"
        )
    if tool_call_log_path is None:
        tool_call_log_path = result_path.parent / f"{result_path.stem}.tool_calls.jsonl"

    if (context_path.exists() or problem_path.exists()) and not overwrite_problem:
        raise FileExistsError(
            f"Exported problem files already exist in {run_dir}. Use --overwrite-problem to regenerate them."
        )

    train_data = {feature: train[:, i + 1].astype(float) for i, feature in enumerate(features)}
    train_data[target] = train[:, 0].astype(float)
    np.savez(context_path, data=train_data, target=target)

    problem_description = build_problem_description(problem)
    np.savez(
        problem_path,
        dataset_identifier=problem.dataset_identifier,
        equation_id=problem.equation_idx,
        symbols=np.array(problem.symbols, dtype=object),
        symbol_descs=np.array(problem.symbol_descs, dtype=object),
        symbol_properties=np.array(problem.symbol_properties, dtype=object),
        problem_description=problem_description,
        train_X=train[:, 1:].astype(float),
        train_y=train[:, 0].astype(float),
        context_path=str(context_path),
        result_path=str(result_path),
    )

    result = create_initial_result(
        result_path=result_path,
        problem=problem,
        problem_path=problem_path,
        context_path=context_path,
        tool_call_log_path=tool_call_log_path,
        agent=agent,
        overwrite_result=overwrite_result,
    )
    manifest = {
        "dataset_identifier": problem.dataset_identifier,
        "equation_id": problem.equation_idx,
        "target": target,
        "features": features,
        "num_train": int(len(train)),
        "num_test": int(len(test)),
        "num_ood": int(0 if ood is None else len(ood)),
        "problem_path": problem_path,
        "context_path": context_path,
        "manifest_path": manifest_path,
        "result_path": result_path,
        "tool_call_log_path": tool_call_log_path,
        "problem_description": problem_description,
        "initial_result": result,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one LLM-SRBench problem for a coding-agent baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SPLITS), help="Benchmark dataset.")
    parser.add_argument("--problem-name", required=True, help="Equation/problem id.")
    parser.add_argument("--agent", default="coding_agent", help="Agent name used in the result filename.")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[2] / "data" / "llm-srbench-data"),
        help="LLM-SRBench data root.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory under which public files are written.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory for the initial result JSON.")
    parser.add_argument("--result-path", default=None, help="Exact path for the initial result JSON. Overrides --result-dir naming.")
    parser.add_argument("--tool-call-log-path", default=None, help="Exact JSONL path for tool-call records.")
    parser.add_argument("--overwrite-problem", action="store_true", help="Regenerate public problem files if they already exist.")
    parser.add_argument("--overwrite-result", action="store_true", help="Overwrite an existing initial result JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    problem = select_problem(args.dataset, args.problem_name, args.data_root)
    manifest = export_problem(
        problem,
        output_dir=Path(args.output_dir),
        result_dir=Path(args.result_dir),
        agent=args.agent,
        result_path=Path(args.result_path) if args.result_path else None,
        tool_call_log_path=Path(args.tool_call_log_path) if args.tool_call_log_path else None,
        overwrite_problem=args.overwrite_problem,
        overwrite_result=args.overwrite_result,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
