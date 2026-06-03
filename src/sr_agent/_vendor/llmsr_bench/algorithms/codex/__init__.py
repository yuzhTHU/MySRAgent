# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Codex CLI coding-agent algorithm for llmsr_bench."""
from __future__ import annotations
import os
import re
import sys
import time
import json
import shlex
import shutil
import logging
import argparse
import selectors
import subprocess
import nd2py as nd
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime
from sr_agent.tools import BaseTool
from sr_agent.utils import tag2ansi
from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[6]
_README_TEMPLATE_PATH = _PACKAGE_DIR / "readme_template.md"
_CALL_TOOL_TEMPLATE_PATH = _PACKAGE_DIR / "call_tool_template.py"

os.environ.setdefault("HF_HOME", "/tmp/sr_agent_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/sr_agent_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr_agent_mplconfig")
for _cache_dir in ("HF_HOME", "HF_DATASETS_CACHE", "MPLCONFIGDIR"):
    Path(os.environ[_cache_dir]).mkdir(parents=True, exist_ok=True)

__all__ = ["update_parser", "run"]
_logger = logging.getLogger(f"sr_agent.{__name__}")


def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--codex_cmd", default=os.environ.get("CODEX_CMD"), help="Full Codex command prefix, e.g. 'npx --yes @openai/codex@latest'. Overrides --codex_bin.")
    parser.add_argument("--codex_bin", default=os.environ.get("CODEX_BIN", "codex"), help="Codex executable used when --codex_cmd is unset.")
    parser.add_argument("--codex_model", default=os.environ.get("CODEX_MODEL", "gpt-5.5"), help="Model passed to Codex CLI.")
    parser.add_argument("--codex_timeout_seconds", default=int(os.environ.get("CODEX_TIMEOUT_SECONDS", "900")), type=int, help="Per-problem Codex wall-clock timeout.")
    parser.add_argument("--codex_progress_interval", default=int(os.environ.get("CODEX_PROGRESS_INTERVAL", "30")), type=int, help="Seconds between Codex progress log lines. Use 0 to disable.")
    parser.add_argument("--codex_echo_events", action='store_true', default=True, help="Print raw Codex JSONL events while saving them.")
    parser.add_argument("--codex_sandbox", default=os.environ.get("CODEX_SANDBOX", "workspace-write"), help="Sandbox mode passed to Codex CLI.")
    parser.add_argument("--codex_approval_policy", default=os.environ.get("CODEX_APPROVAL_POLICY"), help="Optional Codex config override for approval_policy.")
    parser.add_argument("--codex_extra_args", default=os.environ.get("CODEX_EXTRA_ARGS", ""), type=str, help="Extra arguments inserted before the prompt.")
    parser.add_argument("--codex_overwrite", action='store_true', default=False, help="Overwrite per-problem Codex public files and result JSON.")
    parser.add_argument("--tools", default=BaseTool.all_registered_names, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("--ban_tools", default=[], type=str, nargs='+', help="Optional list of tools to exclude. Default is no excluded tools.")
    return parser


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    # 初始化
    artifacts = export_task(args, task)
    command = build_codex_command(args, artifacts)
    command_for_log = " ".join(shlex.quote(part) for part in command[:-1]) + " <initial_prompt>"
    _logger.note(tag2ansi(
        f"[blue bold][CODEX RUN][reset] [green]{artifacts['problem_name']}[reset]\n"
        f"  [blue]dir:[reset]        [green]{artifacts['problem_dir']}[reset]\n"
        f"  [blue]problem:[reset]    [green]{artifacts['problem_path']}[reset]\n"
        f"  [blue]context:[reset]    [green]{artifacts['context_path']}[reset]\n"
        f"  [blue]result:[reset]     [green]{artifacts['result_path']}[reset]\n"
        f"  [blue]events:[reset]     [green]{artifacts['event_path']}[reset]\n"
        f"  [blue]cmd:[reset]        [green]{command_for_log}[reset]"
    ))

    # 运行
    status = run_codex_command(command, artifacts, args)

    # 后处理
    result = load_result_json(artifacts["result_path"])
    result['token_usage'] = latest_usage_from_codex_events(artifacts["event_path"])
    result['tool_call_count'] = count_jsonl(artifacts["tool_call_log_path"])
    result['end_time'] = (end_time := datetime.now()).isoformat()
    result['duration_seconds'] = (end_time - artifacts["start_time"]).total_seconds()
    if expression := (result.pop("discovered_expression", None) or result.pop("formula", None)):
        result['discovered_expression'] = expression
    elif expression := best_formula_from_tool_calls(artifacts["tool_call_log_path"]):
        result['discovered_expression'] = expression
        result['notes'] = (result.get('notes') or '') + "\nFallback: selected the lowest-mse formula from tool-call records."
    elif status != 0:
        raise RuntimeError(f"Codex exited with status {status} and did not write discovered_expression. See {artifacts['event_path']}")
    else:
        raise ValueError(f"Codex did not discover an expression. See {artifacts['event_path']} for details.")
    artifacts["result_path"].write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")

    # 返回
    f = nd.parse(expression.strip().replace("^", "**").replace("np.", "").replace("math.", ""))
    target = task.symbols[0]
    features = task.symbols[1:]
    constants = {}
    for var in f.iter_preorder():
        if not isinstance(var, nd.Variable) or var.name in features:
            pass
        elif var.name.lower() == 'pi':
            constants[var.name] = np.pi
        elif var.name.lower() == 'e':
            constants[var.name] = np.e
        else:
            raise ValueError(f"Unknown variable '{var.name}' in discovered expression that is not in features. Please ensure all variables are either features or known constants like pi and e.")

    def predict(X: np.ndarray) -> np.ndarray:
        pred_data = {feat: X[:, i] for i, feat in enumerate(features)}
        pred_data[target] = np.zeros(len(X))  # 占位，不会被使用
        pred_data |= constants # 将常数也加入数据字典，供表达式求值使用
        return f.eval(pred_data).flatten()
    
    return SRResult(predict=predict, expression=expression)


def export_task(args: argparse.Namespace, task: SEDTask) -> dict[str, Path]:
    target = task.symbols[0]
    features = task.symbols[1:]
    lines = []
    for sym, desc, prop in zip(task.symbols, task.symbol_descs, task.symbol_properties):
        kind = {"O": "Output", "V": "Input Variable"}.get(prop, "Unknown")
        lines.append(f"{sym} ({kind}): {desc}")
    problem_description = "\n".join(lines)
    start_time = datetime.now()

    # 初始化目录
    save_name = sanitize_filename(task.name)
    if args.save_path is not None:
        problem_dir = Path(args.save_path) / "experiments" / f"{save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        problem_dir = Path("/tmp") / "sr_agent_codex" / f"{save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    readme_path        = problem_dir / "README.md"          # Agent 本地操作说明
    manifest_path      = problem_dir / "manifest.json"      # 存储运行元数据
    problem_path       = problem_dir / "problem.json"       # 存储问题描述
    call_tool_path     = problem_dir / "call_tool.py"       # Agent 本地工具入口
    context_path       = problem_dir / "context.npz"        # 存储用于 call_tool 的上下文
    result_path        = problem_dir / "result.json"        # 存储运行结果
    final_path         = problem_dir / "final_message.txt"  # 记录 Codex 输出的最终消息
    event_path         = problem_dir / "codex_events.jsonl" # 记录 Codex 输出的事件流
    tool_call_log_path = problem_dir / "tool_calls.jsonl"   # 记录工具调用日志
    if not args.codex_overwrite and any(path.exists() for path in (context_path, problem_path, result_path)):
        raise FileExistsError(f"Codex artifacts already exist in {problem_dir}. Use --codex-overwrite to regenerate them.")

    manifest_path.write_text(json.dumps({
        "equation_id": task.name,
        "target": target,
        "features": features,
        "symbols": task.symbols,
        "symbol_descs": task.symbol_descs,
        "symbol_properties": task.symbol_properties,
        "num_train": int(len(task.train_y)),
        "problem_dir": str(problem_dir.absolute()),
        "problem_path": str(problem_path.relative_to(problem_dir)),
        "context_path": str(context_path.relative_to(problem_dir)),
        "manifest_path": str(manifest_path.relative_to(problem_dir)),
        "result_path": str(result_path.relative_to(problem_dir)),
        "readme_path": str(readme_path.relative_to(problem_dir)),
        "call_tool_path": str(call_tool_path.relative_to(problem_dir)),
        "tool_call_log_path": str(tool_call_log_path.relative_to(problem_dir)),
        "problem_description": problem_description,
    }, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

    problem_path.write_text(json.dumps({
        "symbols": task.symbols,
        "symbol_descs": task.symbol_descs,
        "symbol_properties": task.symbol_properties,
        "problem_description": problem_description,
    }, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    
    np.savez(context_path, **{
        'data': {feature: task.train_X[:, i].astype(float) for i, feature in enumerate(features)} | {target: task.train_y.astype(float)},
        'target': target,
    })

    readme_path.write_text((
        _README_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("<WORK_DIR>", str(problem_dir.absolute()))
        .replace("<PROBLEM_PATH>", str(problem_path.relative_to(problem_dir)))
        .replace("<MANIFEST_PATH>", str(manifest_path.relative_to(problem_dir)))
        .replace("<RESULT_PATH>", str(result_path.relative_to(problem_dir)))
        .replace("<CALL_TOOL_PATH>", str(call_tool_path.relative_to(problem_dir)))
        .replace("<CONTEXT_PATH>", str(context_path.relative_to(problem_dir)))
        .replace("<README_PATH>", str(readme_path.relative_to(problem_dir)))
        .replace("<TIMEOUT_SECONDS>", str(args.codex_timeout_seconds))
    ), encoding="utf-8")

    call_tool_path.write_text((
        _CALL_TOOL_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("<WORK_DIR>", str(problem_dir.absolute()))
        .replace("<CONTEXT_PATH>", str(context_path.relative_to(problem_dir)))
        .replace("<TOOL_CALL_LOG_PATH>", str(tool_call_log_path.relative_to(problem_dir)))
        .replace("<ENABLED_TOOLS>", ','.join(sorted(set(args.tools) - set(args.ban_tools))))
    ), encoding="utf-8")

    result_path.write_text(json.dumps({
        "equation_id": task.name,
        "start_time": start_time.isoformat(),
        "status": "started",
        "end_time": None,
        "duration_seconds": None,
        "tool_call_count": None,
        "token_usage": None,
        "discovered_expression": None,
        "notes": None,
    }, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")

    return {
        "problem_name": task.name,
        "problem_dir": problem_dir,
        "context_path": context_path,
        "problem_path": problem_path,
        "manifest_path": manifest_path,
        "result_path": result_path,
        "readme_path": readme_path,
        "call_tool_path": call_tool_path,
        "tool_call_log_path": tool_call_log_path,
        "event_path": event_path,
        "final_path": final_path,
        "start_time": start_time,
    }


def build_codex_command(args: argparse.Namespace, artifacts: dict[str, Path]) -> list[str]:
    prefix = codex_command_prefix(args)
    extra_args = shlex.split(args.codex_extra_args)
    approval_args = []
    if args.codex_approval_policy:
        approval_args = ["-c", f"approval_policy={json.dumps(args.codex_approval_policy)}"]
    return [
        *prefix,
        "exec",
        "--json",
        "-C", str(artifacts["problem_dir"]),
        "-s", args.codex_sandbox, *approval_args,
        "-m", args.codex_model,
        "-o", str(artifacts["final_path"]),
        *extra_args,
        artifacts["readme_path"].read_text(encoding="utf-8"),
    ]


def run_codex_command(command: list[str], artifacts, args: argparse.Namespace) -> int:
    result_path = artifacts["result_path"]
    tool_call_log_path = artifacts["tool_call_log_path"]
    start = time.monotonic()
    deadline = start + args.codex_timeout_seconds
    next_progress = start + args.codex_progress_interval
    event_count = 0

    with artifacts["event_path"].open("w", encoding="utf-8") as event_file:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT, # 这里不能改成 problem_dir, 因为 problem_dir 是相对于 _REPO_ROOT 的路径
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    event_file.write(json.dumps({"type": "bench.timeout", "timeout_seconds": args.codex_timeout_seconds}, ensure_ascii=False) + "\n")
                    event_file.flush()
                    return 124

                for key, _ in selector.select(timeout=1):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    event_file.write(line)
                    event_file.flush()
                    event_count += 1
                    if args.codex_echo_events:
                        _logger.info(line.rstrip())

                if args.codex_progress_interval > 0 and now >= next_progress:
                    elapsed = int(now - start)
                    tool_count = count_jsonl(tool_call_log_path)
                    _logger.info(tag2ansi(
                        "[yellow bold][CODEX WAIT][reset] "
                        f"elapsed={elapsed}s events={event_count} "
                        f"tool_calls={tool_count if tool_count is not None else 0} "
                        f"status=[blue]{result_status(result_path)}[reset]"
                    ))
                    next_progress = now + args.codex_progress_interval

            for line in process.stdout:
                event_file.write(line)
                event_file.flush()
                event_count += 1
                if args.codex_echo_events:
                    _logger.info(line.rstrip())
        finally:
            selector.close()

    return int(process.returncode or 0)


def latest_usage_from_codex_events(event_path: Path) -> dict[str, Any] | None:

    def find_usage(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            keys = set(value)
            if {"input_tokens", "output_tokens"} & keys or "total_tokens" in keys:
                return value
            for child in value.values():
                found = find_usage(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_usage(child)
                if found:
                    return found
        return None

    usage = None
    if not event_path.exists():
        return None
    for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate := find_usage(event):
            usage = candidate
    return usage


def load_result_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def result_status(path: Path) -> str:
    if not (result := load_result_json(path)):
        return "result JSON missing or unreadable"
    else:
        status = result.get("status") or "unknown"
        expression = result.get("discovered_expression", "(No Expression)")
        return f"[{status}] {expression}"


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def best_formula_from_tool_calls(path: Path) -> str | None:
    best_record = None
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
            result = record["tool_call_result"]["result"]
            formula = result["formula"]
            metrics = result["metrics"]
            mse = float(metrics["mse"])
        except Exception:
            continue
        if best_record is None or mse < best_record["mse"]:
            best_record = {"formula": formula, **metrics}
    return best_record["formula"] if best_record else None


def codex_command_prefix(args: argparse.Namespace) -> list[str]:
    value = args.codex_cmd or args.codex_bin
    prefix = shlex.split(value) if isinstance(value, str) else list(value)
    if not prefix:
        raise ValueError("Codex command is empty. Set --codex_cmd or --codex_bin.")
    return prefix


def sanitize_filename(value: str) -> str:
    return re.sub(r'[ <>:"/\\|?*\x00-\x1f]', "_", value.strip())[:255] or "unnamed"
