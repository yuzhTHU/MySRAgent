# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Codex CLI coding-agent algorithm for llmsr_bench."""
from __future__ import annotations
import os
import re
import sys
import time
import signal
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
    parser.add_argument("--codex_finalize_timeout_seconds", default=int(os.environ.get("CODEX_FINALIZE_TIMEOUT_SECONDS", "60")), type=int, help="Wall-clock timeout for the finalization pass (second Codex call). 60s only allows a fast memory-based submission; reading logs / re-analysis times out and counts as missing.")
    parser.add_argument("--no_codex_finalize", action='store_true', default=False, help="Disable the finalization pass: a second Codex call that extracts the best formula from the main pass's exploration log when result.json was not completed.")
    parser.add_argument("--tools", default=BaseTool.all_registered_names, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("--ban_tools", default=[], type=str, nargs='+', help="Optional list of tools to exclude. Default is no excluded tools.")
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider used for the symbolic-accuracy equivalence judge.")
    parser.add_argument("--llm_model", default="qwen/qwen3.5-flash-02-23", help="LLM model used for the symbolic-accuracy equivalence judge.")
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

    # 二阶段 finalize pass：主 agent 未以 completed 提交有效公式时
    # （典型：DeepSeek 过度思考导致超时，result.json 只有 in_progress 基线），
    # 另起一个 Codex 实例，从主 pass 的探索日志中提取最佳公式并写入 result.json。
    result = load_result_json(artifacts["result_path"])
    main_submitted = result.get("status") == "completed" and (result.get("discovered_expression") or result.get("formula"))
    if not main_submitted and not args.no_codex_finalize:
        _logger.note(tag2ansi(
            f"[blue bold][CODEX FINALIZE][reset] main pass did not complete a submission "
            f"(status=[blue]{result.get('status', 'unknown')}[reset]); "
            f"starting finalization pass with a fresh Codex instance..."
        ))
        finalize_status = run_finalize_pass(args, artifacts)
        _logger.note(tag2ansi(
            f"[blue bold][CODEX FINALIZE][reset] finalization pass exited with status "
            f"[green]{finalize_status}[reset]; result: [blue]{result_status(artifacts['result_path'])}[reset]"
        ))
        result = load_result_json(artifacts["result_path"])

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
    finalize_event_path = problem_dir / "codex_finalize_events.jsonl" # 记录 finalize pass 的事件流
    finalize_message_path = problem_dir / "finalize_message.txt"     # 记录 finalize pass 的最终消息
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
        "finalize_event_path": finalize_event_path,
        "finalize_message_path": finalize_message_path,
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


_FINALIZE_PROMPT = """You are the finalization pass of a symbolic-regression benchmark run. A previous coding agent already explored this problem for the full time budget but never wrote a final formula to `result.json`. Your only job is to recover the best formula the previous agent found and write it to `result.json`. This is not optional: the run is judged by `result.json`, and it currently has no valid `discovered_expression`.

The problem directory contains the previous agent's full exploration log:
- `codex_events.jsonl`: every command the previous agent ran and its output. It very likely contains candidate formulas and their evaluation scores (e.g. `R²=`, `RMSE`, `r2`, `Reformulated`, `best formula`) printed during exploration.
- `problem.json` and `manifest.json`: problem description and variable metadata (the first symbol is the target; the rest are input features).
- `context.npz`: training data. Load with `np.load('context.npz', allow_pickle=True)`; the `data` key holds a dict of feature arrays plus the target array.
- `README.md`: the original task instructions.
- `result.json`: the only file you are allowed to write.

Follow these steps in order:
1. Immediately scan `codex_events.jsonl` and extract the best candidate formula the previous agent found, i.e. the one with the highest R² / lowest error. Search for lines containing `R²`, `RMSE`, `r2`, `formula`, or `Reformulated`.
2. If the formula is not obvious or looks wrong, refuse to submit: do not create or modify `result.json`, and exit.
3. Otherwise, write `result.json` now:

{"discovered_expression": "<expr>", "status": "completed", "notes": "<short note>"}

The expression must use only the feature variables listed in `manifest.json` plus `pi`/`e` constants. Do not over-search: extracting the previous agent's best formula and submitting it is far better than submitting nothing."""


_RESUME_FINALIZE_PROMPT = """Your earlier exploration of this symbolic-regression problem was cut off by the time budget before you wrote a final formula to `result.json`. The run is judged by `result.json`, and it currently holds only your initial `in_progress` baseline, so the run counts as FAILED unless you submit now.

Your task, and nothing else: look at your exploration history (your memory and `codex_events.jsonl`), identify the best formula it contains, and submit it to `result.json`. Do not run any verification, calculation, or data analysis. Do not start a new search. If you cannot identify a formula you consider credible, refuse to submit: do not create or modify `result.json`, and exit.

Update `result.json` now: preserve its existing fields and fill `discovered_expression` and `status` ("completed"), optionally `notes`:

{"discovered_expression": "<expr>", "status": "completed", "notes": "<short note>"}

The expression must use only the feature variables listed in `manifest.json` plus `pi`/`e` constants. Submitting a formula is success; searching further is failure."""


def get_thread_id_from_events(event_path: Path) -> str | None:
    """从主 pass 事件流解析 codex 会话 thread_id（供 finalize pass 续接会话）。"""
    try:
        with event_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    return event.get("thread_id")
    except OSError:
        return None
    return None


def _codex_session_meta(session_id: str) -> dict[str, Any]:
    """在 ~/.codex/sessions 下定位该会话的 rollout 文件，读取 session_meta 元信息。"""
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    try:
        for rollout in codex_home.glob(f"sessions/**/rollout-*-{session_id}.jsonl"):
            with rollout.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "session_meta":
                        return event.get("payload") or {}
    except OSError:
        return {}
    return {}


def build_finalize_command(args: argparse.Namespace, artifacts: dict[str, Path], thread_id: str | None = None) -> list[str]:
    prefix = codex_command_prefix(args)
    approval_args = []
    if args.codex_approval_policy:
        approval_args = ["-c", f"approval_policy={json.dumps(args.codex_approval_policy)}"]
    if thread_id:
        # resume 续接主 pass 会话，继承其记忆与判断（历史最优公式的判定与主 agent 一致）。
        # resume 不支持 -p/--profile、-s、-C：剥离 profile 参数；provider 从会话元信息
        # 恢复（找不到则依赖会话继承，sandbox/cwd 由会话自带，与主 pass 一致）。
        extra_args, skip_next = [], False
        for part in shlex.split(args.codex_extra_args):
            if skip_next:
                skip_next = False
                continue
            if part in ("-p", "--profile"):
                skip_next = True
                continue
            if part.startswith("--profile="):
                continue
            extra_args.append(part)
        provider_args = []
        meta = _codex_session_meta(thread_id)
        if meta.get("model_provider"):
            provider_args = ["-c", f'model_provider="{meta["model_provider"]}"']
        return [
            *prefix,
            "exec", "resume",
            "--json",
            "-m", args.codex_model,
            *provider_args,
            "-o", str(artifacts["finalize_message_path"]),
            *approval_args,
            *extra_args,
            thread_id,
            _RESUME_FINALIZE_PROMPT,
        ]
    extra_args = shlex.split(args.codex_extra_args)
    return [
        *prefix,
        "exec",
        "--json",
        "-C", str(artifacts["problem_dir"]),
        "-s", args.codex_sandbox, *approval_args,
        "-m", args.codex_model,
        "-o", str(artifacts["finalize_message_path"]),
        *extra_args,
        _FINALIZE_PROMPT,
    ]


def run_finalize_pass(args: argparse.Namespace, artifacts: dict[str, Path]) -> int:
    thread_id = get_thread_id_from_events(artifacts["event_path"])
    if thread_id:
        _logger.note(tag2ansi(
            f"[blue bold][CODEX FINALIZE][reset] resuming main-pass session "
            f"[green]{thread_id}[reset] to recover its best formula..."
        ))
    else:
        _logger.note(tag2ansi(
            f"[yellow bold][CODEX FINALIZE][reset] no thread.started in events; "
            f"falling back to a cold-start Codex instance"
        ))
    command = build_finalize_command(args, artifacts, thread_id)
    command_for_log = " ".join(shlex.quote(part) for part in command[:-1]) + " <finalize_prompt>"
    _logger.info(tag2ansi(f"[blue bold][CODEX FINALIZE][reset] {command_for_log}"))
    status = run_codex_command(
        command, artifacts, args,
        timeout_seconds=args.codex_finalize_timeout_seconds,
        event_path=artifacts["finalize_event_path"],
    )
    if status != 0 and thread_id:
        # resume 失败重试一次：典型原因是主 pass 超时被强杀后 codex 会话写锁残留，
        # resume 因 thread-store conflict 立即退出；codex 退出时会释放锁，
        # 等待片刻后重试可成功（v6 实测：失败后锁目录已清空）。
        _logger.note(tag2ansi(
            f"[yellow bold][CODEX FINALIZE][reset] resume failed (status={status}); "
            f"waiting 2s and retrying once..."
        ))
        time.sleep(2)
        status = run_codex_command(
            command, artifacts, args,
            timeout_seconds=args.codex_finalize_timeout_seconds,
            event_path=artifacts["finalize_event_path"].with_name("codex_finalize_retry_events.jsonl"),
        )
    # 统计分层（学长的三分类设计）：主 pass 自提交 = completed；resume 补交 = resume；
    # resume 也搞不定 = missing。resume 成功时把 status 从 completed 改为 resume，
    # 实验结束后扫 experiments/*/result.json 即可区分三类题。
    result_path = artifacts["result_path"]
    result = load_result_json(result_path)
    if status == 0 and result.get("status") == "completed" and (
        result.get("discovered_expression") or result.get("formula")
    ):
        result["status"] = "resume"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
        _logger.note(tag2ansi(
            f"[blue bold][CODEX FINALIZE][reset] resume submitted a formula; "
            f"marking result status as [green]resume[/green]."
        ))
        return 0
    # resume 也搞不定：把 status 标记为 missing 存进 result.json，
    # 供后续统计定位与更优模型补跑。
    result["status"] = "missing"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    _logger.note(tag2ansi(
        f"[yellow bold][CODEX FINALIZE][reset] finalize failed (status={status}); "
        f"marking result status as [red]missing[/red]."
    ))
    return 1


def _kill_group(process: subprocess.Popen, sig: int) -> None:
    """向 codex 进程组发送信号；进程组已消失（进程恰好自然退出）时静默忽略。"""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_codex_command(command: list[str], artifacts, args: argparse.Namespace, timeout_seconds: int | None = None, event_path: Path | None = None) -> int:
    result_path = artifacts["result_path"]
    tool_call_log_path = artifacts["tool_call_log_path"]
    timeout_seconds = args.codex_timeout_seconds if timeout_seconds is None else timeout_seconds
    event_path = artifacts["event_path"] if event_path is None else event_path
    start = time.monotonic()
    deadline = start + timeout_seconds
    next_progress = start + args.codex_progress_interval
    event_count = 0
    # 提交即止：agent 将 result.json 置为 completed 后提前结束，避免提交后继续消耗时间与 token。
    baseline_mtime = result_path.stat().st_mtime if result_path.exists() else 0.0

    with event_path.open("w", encoding="utf-8") as event_file:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT, # 这里不能改成 problem_dir, 因为 problem_dir 是相对于 _REPO_ROOT 的路径
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # 独立进程组：信号用 killpg 全组发送，保证到达 codex 二进制
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    # 先 SIGINT 让 codex 优雅退出（清理会话写锁），避免强杀残留
                    # thread-store 写锁导致后续 finalize resume 立即失败。
                    # 必须 killpg：进程链为 npx -> codex.js(node shim) -> 二进制，
                    # 单发信号只打到 npx，codex.js 收不到就不会转发，优雅退出落空。
                    _kill_group(process, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _kill_group(process, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            _kill_group(process, signal.SIGKILL)
                            process.wait()
                    event_file.write(json.dumps({"type": "bench.timeout", "timeout_seconds": timeout_seconds}, ensure_ascii=False) + "\n")
                    event_file.flush()
                    return 124

                # 提交即止：result.json 被置为 completed 后提前结束（见上方 baseline_mtime 注释）。
                try:
                    if result_path.stat().st_mtime > baseline_mtime:
                        with result_path.open("r", encoding="utf-8") as rf:
                            _r = json.load(rf)
                        if _r.get("status") == "completed":
                            _logger.info(tag2ansi(
                                f"[green bold][CODEX DONE][reset] result.json marked completed; "
                                f"stopping early (elapsed={int(now - start)}s)"
                            ))
                            # 终止 codex 进程组，避免提交后孤儿进程继续消耗时间与 token。
                            _kill_group(process, signal.SIGINT)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                _kill_group(process, signal.SIGKILL)
                                process.wait()
                            return 0
                except (OSError, json.JSONDecodeError):
                    pass

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
    if usage is not None:
        return usage
    # codex 0.152 的 stdout 事件流不含 token 统计（v8 实测 token_usage 恒为 None）：
    # 回退到 codex 会话 rollout 文件，读取最后一条 token_count 的累计用量
    # （口径含 finalize/resume 回合，与之前手动统计一致）。
    thread_id = get_thread_id_from_events(event_path)
    if not thread_id:
        return None
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    try:
        for rollout in codex_home.glob(f"sessions/**/rollout-*-{thread_id}.jsonl"):
            for line in rollout.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip() or not line.strip().startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = event.get("payload") or {}
                if event.get("type") == "event_msg" and payload.get("type") == "token_count":
                    usage = (payload.get("info") or {}).get("total_token_usage")
    except OSError:
        pass
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
        # 文件不存在 = 没有任何工具调用记录（no-tool 模式下必然如此），计数为 0。
        return 0
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
            split_results = result["data_split_results"]
            split_name = "validation" if "validation" in split_results else "train"
            metrics = split_results[split_name]["metrics"]
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
