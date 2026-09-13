# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""从 codex 会话 rollout 补全 result.json 中缺失的 token_usage / tool_call_count 字段。

背景：v8/v10 早期版本的 latest_usage_from_codex_events 无法从 stdout 事件流
提取 token 统计（codex 0.152 的 --json 流不含 token_count 事件），导致
result.json 的 token_usage 恒为 null。token 的权威数据在
~/.codex/sessions/**/rollout-*-<thread_id>.jsonl 里，本脚本补全它。

用法:
    venv/bin/python scripts/backfill_token_usage.py <experiments_dir>
例如:
    venv/bin/python scripts/backfill_token_usage.py \
        logs/bench_sr_agent/codex_flash_no-tool_anonymize_v10/experiments

只更新 token_usage / tool_call_count 两个字段，其余字段原样保留；
token_usage 已有值的文件跳过。
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path


def find_thread_id(event_path: Path) -> str | None:
    try:
        with event_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    return event.get("thread_id")
    except OSError:
        pass
    return None


def total_token_usage_from_rollout(thread_id: str) -> dict | None:
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    usage = None
    try:
        for rollout in codex_home.glob(f"sessions/**/rollout-*-{thread_id}.jsonl"):
            with rollout.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("{"):
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


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    experiments = Path(sys.argv[1])
    if not experiments.is_dir():
        print(f"目录不存在: {experiments}")
        return 1

    updated = skipped = failed = 0
    for result_path in sorted(experiments.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"[跳过] {result_path.parent.name}: result.json 无法读取")
            failed += 1
            continue
        changed = False
        if result.get("token_usage") is None:
            thread_id = find_thread_id(result_path.parent / "codex_events.jsonl")
            usage = total_token_usage_from_rollout(thread_id) if thread_id else None
            if usage:
                result["token_usage"] = usage
                changed = True
        if result.get("tool_call_count") is None:
            # no-tool 模式下 tool_calls.jsonl 从不创建，调用次数即 0
            result["tool_call_count"] = 0
            changed = True
        if changed:
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            updated += 1
            print(f"[补全] {result_path.parent.name}: token_usage={'yes' if result.get('token_usage') else 'no'}")
        else:
            skipped += 1
    print(f"\n完成: 补全 {updated} 个, 跳过(已完整) {skipped} 个, 失败 {failed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
