# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Thin wrapper around the sr-agent tool CLI for coding-agent baselines."""
from __future__ import annotations

import os
import sys
import json
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", "/tmp/sr_agent_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/sr_agent_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr_agent_mplconfig")
for cache_dir in ("HF_HOME", "HF_DATASETS_CACHE", "MPLCONFIGDIR"):
    Path(os.environ[cache_dir]).mkdir(parents=True, exist_ok=True)

from sr_agent.cli.tool import main as tool_main
from sr_agent.tools import BaseTool, ToolCallResult


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_log_path(context_path: str | Path) -> Path | None:
    context_path = Path(context_path)
    manifest_path = context_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    log_path = manifest.get("tool_call_log_path")
    if log_path:
        return Path(log_path)
    result_path = manifest.get("result_path")
    if not result_path:
        return None
    result_path = Path(result_path)
    return result_path.parent / f"{result_path.stem}.tool_calls.jsonl"


def log_tool_call(args, params: dict[str, Any], result: ToolCallResult) -> None:
    log_path = resolve_log_path(args.context)
    if log_path is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": args.tool,
        "context_path": str(args.context),
        "params": params,
        "tool_call_result": {
            "ok": result.ok,
            "result": result.result,
            "result_str": result.result_str,
            "meta_data": result.meta_data,
        },
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, allow_nan=True, default=json_default)
        f.write("\n")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "list":
        for name, tool_cls in BaseTool.REGISTRY_DICT.items():
            description = (tool_cls.metadata.description or "").strip().splitlines()[0]
            print(f"{name}: {description}")
        raise SystemExit(0)
    tool_main(argv, on_tool_result=log_tool_call)


if __name__ == "__main__":
    main()
