#!/usr/bin/env python
# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Run one fresh Codex CLI agent per symbolic-regression benchmark problem."""
from __future__ import annotations
import os
import re
import sys
import time
import json
import stat
import shlex
import shutil
import logging
import argparse
import selectors
import subprocess
from typing import Any
from pathlib import Path
from datetime import datetime
from socket import gethostname


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", "/tmp/sr_agent_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/sr_agent_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr_agent_mplconfig")
for cache_dir in ("HF_HOME", "HF_DATASETS_CACHE", "MPLCONFIGDIR"):
    Path(os.environ[cache_dir]).mkdir(parents=True, exist_ok=True)

from bench_sr_agent import load_problems
from sr_agent.utils import setup_logging, tag2ansi, seed_all, add_minus_flags, add_negation_flags


SCRIPT_NAME = Path(__file__).stem  # bench_coding_agent
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")
DEFAULT_LOG_ROOT = Path("logs/baseline/coding_agent")
DEFAULT_DATA_ROOT = Path("data/llm-srbench-data")


def sanitize_filename(value: str) -> str:
    return re.sub(r'[ <>:"/\\|?*\x00-\x1f]', "_", value.strip())[:255] or "unnamed"


def save_args(args: argparse.Namespace, args_path: Path) -> None:
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists():
            i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    with args_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)


def list_problem_names(dataset: str, data_root: Path) -> list[str]:
    return [problem.equation_idx for problem in load_problems(dataset, str(data_root))]


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def codex_command_prefix(args: argparse.Namespace) -> list[str]:
    value = args.codex_cmd or args.codex_bin
    prefix = shlex.split(value) if isinstance(value, str) else list(value)
    if not prefix:
        raise ValueError("Codex command is empty. Set --codex_cmd or --codex_bin.")
    return prefix


def format_codex_command(prefix: list[str]) -> str:
    resolved = shutil.which(prefix[0])
    command = " ".join(shlex.quote(part) for part in prefix)
    if resolved:
        return f"{command} ({resolved})"
    return command


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


def latest_usage_from_codex_events(event_path: Path) -> dict[str, Any] | None:
    usage = None
    if not event_path.exists():
        return None
    for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = find_usage(event)
        if candidate:
            usage = candidate
    return usage


def result_status(result_path: Path) -> str:
    if not result_path.exists():
        return "no results.json yet"
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return "results.json unreadable"
    status = data.get("status") or "unknown"
    expression = data.get("discovered_expression")
    if expression:
        return f"{status}; discovered_expression={expression}"
    return str(status)


def load_result_json(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def problem_artifacts(
    result_path: Path,
    tool_calls_path: Path,
    session_path: Path,
    commands_path: Path,
    event_path: Path,
    final_path: Path,
) -> list[Path]:
    return [
        path
        for path in (result_path, tool_calls_path, session_path, commands_path, event_path, final_path)
        if path.exists()
    ]


def update_result_metadata(
    result_path: Path,
    event_path: Path,
    tool_calls_path: Path,
    commands_path: Path,
) -> None:
    if not result_path.exists():
        return

    mode = result_path.stat().st_mode
    result_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        result_path.chmod(mode)
        return

    usage = latest_usage_from_codex_events(event_path)
    if usage:
        data["token_usage"] = usage
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_output_tokens",
            "cached_input_tokens",
        ):
            if key in usage:
                data[key] = usage[key]
        if "total_tokens" not in usage and ("input_tokens" in usage or "output_tokens" in usage):
            data["total_tokens"] = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)

    tool_call_count = count_jsonl(tool_calls_path)
    if tool_call_count is not None:
        data["tool_call_count"] = tool_call_count
    command_count = count_jsonl(commands_path)
    if command_count is not None:
        data["command_count"] = command_count

    data["codex_event_log"] = str(event_path)
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    result_path.chmod(mode)


def log_problem_paths(
    problem_name: str,
    problem_dir: Path,
    event_path: Path,
    result_path: Path,
    tool_calls_path: Path,
    final_path: Path,
) -> None:
    _logger.note(tag2ansi(
        f"[blue bold][RUN][reset] [green]{problem_name}[reset]\n"
        f"  [blue]dir:[reset]        [green]{problem_dir}[reset]\n"
        f"  [blue]events:[reset]     [green]{event_path}[reset]\n"
        f"  [blue]results:[reset]    [green]{result_path}[reset]\n"
        f"  [blue]tool calls:[reset] [green]{tool_calls_path}[reset]\n"
        f"  [blue]final msg:[reset]  [green]{final_path}[reset]"
    ))


def run_codex_command(
    command: list[str],
    event_path: Path,
    result_path: Path,
    tool_calls_path: Path,
    args: argparse.Namespace,
) -> int:
    start = time.monotonic()
    deadline = start + args.timeout_seconds
    next_progress = start + args.progress_interval
    event_count = 0

    with event_path.open("w", encoding="utf-8") as event_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
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
                    event_file.write(
                        json.dumps(
                            {"type": "bench.timeout", "timeout_seconds": args.timeout_seconds},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    event_file.flush()
                    return 124

                for key, _ in selector.select(timeout=1):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    event_file.write(line)
                    event_file.flush()
                    event_count += 1
                    if args.echo_events:
                        _logger.info(line.rstrip())

                if args.progress_interval > 0 and now >= next_progress:
                    elapsed = int(now - start)
                    tool_count = count_jsonl(tool_calls_path)
                    _logger.info(tag2ansi(
                        "[yellow bold][WAIT][reset] "
                        f"elapsed={elapsed}s "
                        f"events={event_count} "
                        f"tool_calls={tool_count if tool_count is not None else 0} "
                        f"status=[blue]{result_status(result_path)}[reset]"
                    ))
                    next_progress = now + args.progress_interval

            for line in process.stdout:
                event_file.write(line)
                event_file.flush()
                event_count += 1
                if args.echo_events:
                    _logger.info(line.rstrip())
        finally:
            selector.close()

    return process.returncode


def build_prompt(
    args: argparse.Namespace,
    problem_name: str,
    problem_dir: Path,
    result_path: Path,
    tool_calls_path: Path,
    session_path: Path,
    commands_path: Path,
) -> str:
    fetch_flags = " --overwrite-problem --overwrite-result" if args.overwrite else ""
    return f"""Timed black-box benchmark trial. Follow only baseline/coding_agent/README.md to solve one symbolic-regression problem. Do not start or delegate to any sub-agent.

Problem:
- dataset: {args.dataset}
- problem name: {problem_name}
- agent name: {args.agent_name}

Hard limits:
- Finish within {args.timeout_seconds} seconds.
- If near the limit, submit the best current formula and evaluate it.

Required workflow:
1. Read baseline/coding_agent/README.md.
2. Fetch the problem with:
   python baseline/coding_agent/fetch_problem.py --dataset {args.dataset} --problem-name {problem_name} --agent {args.agent_name} --result-path {result_path} --tool-call-log-path {tool_calls_path}{fetch_flags}
3. Use only generated public files and documented tools. Do not inspect benchmark source/raw data/metadata or ground truth.
4. Follow the required tool workflow in the README, including tool list/schema and candidate evaluation.
5. Fill the generated result JSON and run:
   python baseline/coding_agent/evaluate_result.py --result {result_path} --agent-model "{args.agent_model_label}" --agent {args.agent_name} --agent-provider "{args.agent_provider}" --total-tokens 0 --codex-session-archive {session_path} --codex-commands-archive {commands_path}
   Use --total-tokens 0 if exact token usage is unavailable inside the agent session.
6. Final response must include: commands run, result JSON path, discovered expression, id_metrics, symbolic_acc, and any issues with README clarity or recording.

Do not modify source files.
"""


def run_problem(args: argparse.Namespace, problem_name: str) -> int:
    safe_dataset = sanitize_filename(args.dataset)
    safe_problem = sanitize_filename(problem_name)
    problem_dir = Path(args.save_path) / f"{safe_dataset}_{safe_problem}"
    problem_dir.mkdir(parents=True, exist_ok=True)

    result_path = problem_dir / "results.json"
    tool_calls_path = problem_dir / "tool_calls.jsonl"
    session_path = problem_dir / "codex_session.jsonl"
    commands_path = problem_dir / "codex_commands.jsonl"
    event_path = problem_dir / "codex_events.jsonl"
    final_path = problem_dir / "final_message.txt"

    result = load_result_json(result_path)
    if args.skip_successful and result.get("status") == "evaluated":
        _logger.info(tag2ansi(f"[yellow bold][SKIP][reset] [green]{problem_name}[reset]: evaluated result exists at [green]{result_path}[reset]"))
        return 0

    artifacts = problem_artifacts(result_path, tool_calls_path, session_path, commands_path, event_path, final_path)
    if args.skip_existing and artifacts:
        artifact_list = ", ".join(str(path.name) for path in artifacts)
        _logger.info(tag2ansi(
            f"[yellow bold][SKIP][reset] [green]{problem_name}[reset]: existing artifacts found in "
            f"[green]{problem_dir}[reset] ([blue]{artifact_list}[reset])"
        ))
        return 0

    if args.overwrite and result_path.exists():
        result_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    prompt = build_prompt(args, problem_name, problem_dir, result_path, tool_calls_path, session_path, commands_path)
    approval_args = []
    if args.approval_policy:
        approval_args = ["-c", f"approval_policy={json.dumps(args.approval_policy)}"]

    prefix = codex_command_prefix(args)
    command = [
        *prefix,
        "exec",
        "--json",
        "-C",
        str(REPO_ROOT),
        "-s",
        args.sandbox,
        *approval_args,
        "-m",
        args.codex_model,
        "-o",
        str(final_path),
        *args.codex_extra_args,
        prompt,
    ]

    log_problem_paths(problem_name, problem_dir, event_path, result_path, tool_calls_path, final_path)
    status = run_codex_command(command, event_path, result_path, tool_calls_path, args)

    update_result_metadata(result_path, event_path, tool_calls_path, commands_path)
    if status == 0:
        _logger.note(tag2ansi(f"[green bold][DONE][reset] [green]{problem_name}[reset]: codex exit status {status}; dir=[green]{problem_dir}[reset]"))
    else:
        _logger.warning(tag2ansi(f"[red bold][DONE][reset] [red]{problem_name}[reset]: codex exit status {status}; dir={problem_dir}"))
    return status


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Codex coding-agent symbolic-regression baseline over benchmark problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", "--exp-name", default=os.environ.get("EXP_NAME"), help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", "--log-root", default=os.environ.get("LOG_ROOT", str(DEFAULT_LOG_ROOT)), help="Root directory for logs and run artifacts.")
    parser.add_argument("--save_path", default=None, help="Path to save logs and artifacts. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "-1")), help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (verbose + raise caught exceptions).")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "lsrtransform"))
    parser.add_argument("--data_root", "--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)))
    parser.add_argument("--agent_name", "--agent-name", default=os.environ.get("AGENT_NAME", "codex"))
    parser.add_argument("--agent_provider", "--agent-provider", default=os.environ.get("AGENT_PROVIDER", "openai"))
    parser.add_argument("--codex_cmd", "--codex-cmd", default=os.environ.get("CODEX_CMD"), help="Full Codex command prefix, e.g. 'npx --yes @openai/codex@latest'. Overrides --codex_bin.")
    parser.add_argument("--codex_bin", "--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    parser.add_argument("--codex_model", "--codex-model", default=os.environ.get("CODEX_MODEL", "gpt-5.5"))
    parser.add_argument("--agent_model_label", "--agent-model-label", default=os.environ.get("AGENT_MODEL_LABEL", "GPT-5.5"))
    parser.add_argument("--timeout_seconds", "--timeout-seconds", type=int, default=int(os.environ.get("TIMEOUT_SECONDS", "900")))
    parser.add_argument("--max_problems", "--max-problems", type=int, default=int(os.environ.get("MAX_PROBLEMS", "0")), help="0 means all selected problems.")
    parser.add_argument("--problem_name", "--problem-name", action="append", dest="problem_name_list", help="Problem id to run. Can be passed multiple times.")
    parser.add_argument("--problem_names", "--problem-names", default=os.environ.get("PROBLEM_NAMES"), help="Whitespace-separated problem ids.")
    parser.add_argument("--sandbox", default=os.environ.get("CODEX_SANDBOX", "workspace-write"))
    parser.add_argument(
        "--approval_policy",
        "--approval-policy",
        default=os.environ.get("CODEX_APPROVAL_POLICY"),
        help="Optional Codex config override value for approval_policy. Leave unset to rely on the active Codex config.",
    )
    parser.add_argument("--codex_extra_args", "--codex-extra-args", default=os.environ.get("CODEX_EXTRA_ARGS", ""))
    parser.add_argument("--progress_interval", "--progress-interval", type=int, default=int(os.environ.get("PROGRESS_INTERVAL", "30")), help="Seconds between progress heartbeat lines while Codex is running. Set 0 to disable.")
    parser.add_argument("--echo_events", "--echo-events", action=argparse.BooleanOptionalAction, default=os.environ.get("ECHO_EVENTS", "0") == "1", help="Also print raw Codex JSONL events to the terminal while saving them.")
    parser.add_argument("--skip_existing", "--skip-existing", action=argparse.BooleanOptionalAction, default=os.environ.get("SKIP_EXISTING", "0") == "1", help="Skip a problem if its output directory already contains any run artifact.")
    parser.add_argument("--skip_successful", "--skip-successful", action=argparse.BooleanOptionalAction, default=os.environ.get("SKIP_SUCCESSFUL", "1") != "0", help="Skip a problem if results.json exists and has status='evaluated'.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=os.environ.get("OVERWRITE", "1") != "0")
    parser.add_argument("--dry_run", "--dry-run", action=argparse.BooleanOptionalAction, default=os.environ.get("DRY_RUN", "0") == "1")
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def main(args: argparse.Namespace) -> dict:
    if isinstance(args.codex_extra_args, str):
        args.codex_extra_args = shlex.split(args.codex_extra_args)
    if args.problem_name_list:
        problems = args.problem_name_list
    elif args.problem_names:
        problems = shlex.split(args.problem_names)
    else:
        problems = list_problem_names(args.dataset, args.data_root)

    if args.max_problems:
        problems = problems[: args.max_problems]
    if not problems:
        _logger.error("No problems selected. Check --dataset, --problem_name, --problem_names, and local data files.")
        return {"status": "failed", "failures": 1, "problems": 0}

    run_root = Path(args.save_path)
    codex_prefix = codex_command_prefix(args)
    _logger.note(tag2ansi(
        f"[gray]{'=' * 50}[reset]\n"
        f"[blue bold]Coding-Agent Benchmark[reset]\n"
        f"[blue]Run directory:[reset] [green]{run_root}[reset]\n"
        f"[blue]Dataset:[reset] [green]{args.dataset}[reset]\n"
        f"[blue]Experiment name:[reset] [green]{args.exp_name}[reset]\n"
        f"[blue]Codex command:[reset] [green]{format_codex_command(codex_prefix)}[reset]\n"
        f"[blue]Codex model:[reset] [green]{args.codex_model}[reset]\n"
        f"[blue]Problems:[reset] [green]{len(problems)}[reset]\n"
        f"[gray]{'=' * 50}[reset]"
    ))

    if args.dry_run:
        for problem in problems:
            _logger.info(tag2ansi(f"[blue]DRY RUN:[reset] [green]{problem}[reset]"))
        return {"status": "dry_run", "failures": 0, "problems": len(problems)}

    failures = 0
    for problem in problems:
        status = run_problem(args, problem)
        if status != 0:
            failures += 1

    summary = {"status": "completed" if failures == 0 else "completed_with_failures", "failures": failures, "problems": len(problems)}
    _logger.note(tag2ansi(
        f"[gray]{'=' * 50}[reset]\n"
        f"[blue bold]Completed coding-agent benchmark run[reset]\n"
        f"[blue]Run directory:[reset] [green]{run_root}[reset]\n"
        f"[blue]Problems:[reset] [green]{len(problems)}[reset]\n"
        f"[blue]Failures:[reset] {'[green]0[reset]' if failures == 0 else f'[red]{failures}[reset]'}\n"
        f"[gray]{'=' * 50}[reset]"
    ))
    return summary


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
    save_path = Path(args.save_path) if args.save_path else Path(args.save_dir) / args.exp_name
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
