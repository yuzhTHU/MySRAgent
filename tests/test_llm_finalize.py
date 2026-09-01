# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""LLM finalize 逻辑的离线冒烟测试（不调用任何网络 API）。

用法: venv/bin/python tests/test_llm_finalize.py
"""
from __future__ import annotations
import sys
import json
import argparse
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sr_agent._vendor.llmsr_bench.algorithms.codex import (
    _extract_exploration_text,
    _parse_finalize_json,
    run_finalize_pass,
    _FINALIZE_MAX_LOG_BYTES,
)

PASS = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def make_event_line(ev_type, item_type, item_id, command=None, output=None):
    item = {"id": item_id, "type": item_type}
    if command is not None:
        item["command"] = command
    if output is not None:
        item["aggregated_output"] = output
    return json.dumps({"type": ev_type, "item": item}) + "\n"


# ---------- _extract_exploration_text ----------
print("[1] _extract_exploration_text")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "events.jsonl"
    p.write_text("".join([
        '{"type": "thread.started", "thread_id": "t1"}\n',
        make_event_line("item.started", "command_execution", "c1", "cat problem.json"),
        make_event_line("item.completed", "command_execution", "c1", "cat problem.json", '{"symbols": ["y","x1"]}'),
        make_event_line("item.started", "command_execution", "c2", "python3 -c 'print(1)'"),
        make_event_line("item.completed", "command_execution", "c2", "python3 -c 'print(1)'", "R2=0.99"),
        '{"type": "bench.timeout"}\n',
    ]))
    text = _extract_exploration_text(p)
    check("提取两条命令+输出", "cat problem.json" in text and "R2=0.99" in text)
    check("丢弃簿记事件", "thread.started" not in text and "bench.timeout" not in text)
    check("命令字符串不被按字符拆散", "c a t" not in text)

    p.write_text('{"type": "thread.started", "thread_id": "t1"}\n')
    check("无探索记录返回空串", _extract_exploration_text(p) == "")

    # 截断：36 条 60KB 输出 ≈ 2.1MB，远超 400KB 上限
    lines = []
    for i in range(36):
        lines.append(make_event_line("item.started", "command_execution", f"c{i}", f"cmd{i}"))
        lines.append(make_event_line("item.completed", "command_execution", f"c{i}", f"cmd{i}", ("x" * 60_000) + f" R2={i}"))
    p.write_text("".join(lines))
    text = _extract_exploration_text(p)
    size = len(text.encode("utf-8"))
    check(f"截断后大小受控 ({size/1024:.0f}KB)", size <= _FINALIZE_MAX_LOG_BYTES + 10_000, f"size={size}")
    check("尾部高 R² 输出被保留", "R2=35" in text)
    check("中部输出被丢弃", "R2=18" not in text)
    check("头部输出被保留", "R2=0" in text)

# ---------- _parse_finalize_json ----------
print("[2] _parse_finalize_json")
check("纯 JSON", _parse_finalize_json('{"discovered_expression": "x1"}') == {"discovered_expression": "x1"})
check("markdown 围栏", _parse_finalize_json('```json\n{"discovered_expression": "x1"}\n```') == {"discovered_expression": "x1"})
check("散文包裹 JSON", _parse_finalize_json('最好的公式是 {"discovered_expression": "x1", "status": "completed"}。') == {"discovered_expression": "x1", "status": "completed"})
check("纯散文返回 None", _parse_finalize_json("这不是 JSON") is None)
check("破损 JSON 返回 None", _parse_finalize_json('{"discovered_expression": "x1"') is None)
check("缺字段返回 None", _parse_finalize_json('{"status": "completed"}') is None)
check("null 表达式可解析(拒绝语义)", _parse_finalize_json('{"discovered_expression": null}') == {"discovered_expression": None})

# ---------- run_finalize_pass（mock LLMAPI） ----------
print("[3] run_finalize_pass")


class FakeAPI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        yield reply, None, {}


def make_artifacts(td: Path):
    problem_dir = Path(td)
    (problem_dir / "manifest.json").write_text(json.dumps({"symbols": ["y", "x1", "x2", "x3", "x4"]}))
    result_path = problem_dir / "result.json"
    result_path.write_text(json.dumps({
        "equation_id": "II.test", "start_time": "2026-09-01T00:00:00", "status": "started",
        "end_time": None, "duration_seconds": None, "tool_call_count": None,
        "token_usage": None, "discovered_expression": None, "notes": None,
    }, indent=2))
    events = problem_dir / "codex_events.jsonl"
    events.write_text("".join([
        make_event_line("item.started", "command_execution", "c1", "python3 -c 'print(1)'"),
        make_event_line("item.completed", "command_execution", "c1", "python3 -c 'print(1)'", "y = 142.53*x1*tan(x3): R2=0.99485"),
    ]))
    args = argparse.Namespace(llm_provider="openrouter", llm_model="qwen/qwen3.5-flash-02-23",
                              codex_model="deepseek/deepseek-v4-flash")
    artifacts = {
        "event_path": events,
        "result_path": result_path,
        "finalize_message_path": problem_dir / "finalize_message.txt",
        "manifest_path": problem_dir / "manifest.json",
    }
    return args, artifacts


with tempfile.TemporaryDirectory() as td:
    args, artifacts = make_artifacts(Path(td))
    fake = FakeAPI(['{"discovered_expression": "142.534200*x1*tan(x3)", "status": "completed", "notes": "ok"}'])
    with mock.patch("sr_agent.api.llm_api.LLMAPI.create", return_value=fake):
        status = run_finalize_pass(args, artifacts)
    result = json.loads(artifacts["result_path"].read_text())
    check("成功路径返回 0", status == 0)
    check("result.json 合并保留 equation_id", result["equation_id"] == "II.test")
    check("status 置为 completed", result["status"] == "completed")
    check("表达式写入", result["discovered_expression"] == "142.534200*x1*tan(x3)")
    check("notes 带 finalize 标记", result["notes"].startswith("extracted by llm finalize"))
    check("只调用一次", fake.calls == 1)

with tempfile.TemporaryDirectory() as td:
    args, artifacts = make_artifacts(Path(td))
    fake = FakeAPI(["这不是 JSON", '{"discovered_expression": "x1*tan(x3)"}'])
    with mock.patch("sr_agent.api.llm_api.LLMAPI.create", return_value=fake):
        status = run_finalize_pass(args, artifacts)
    check("首轮失败后重试成功", status == 0 and fake.calls == 2)
    msg = artifacts["finalize_message_path"].read_text()
    check("finalize_message 记录两次尝试", "attempt 1" in msg and "attempt 2" in msg)

with tempfile.TemporaryDirectory() as td:
    args, artifacts = make_artifacts(Path(td))
    fake = FakeAPI(['{"discovered_expression": null, "status": "failed"}'])
    with mock.patch("sr_agent.api.llm_api.LLMAPI.create", return_value=fake):
        status = run_finalize_pass(args, artifacts)
    result = json.loads(artifacts["result_path"].read_text())
    check("拒绝 → 立即失败不重试", status == 1 and fake.calls == 1)
    check("拒绝后 result.json 未被修改", result["status"] == "started" and result["discovered_expression"] is None)

with tempfile.TemporaryDirectory() as td:
    args, artifacts = make_artifacts(Path(td))
    fake = FakeAPI(["散文1", "散文2", "散文3"])
    with mock.patch("sr_agent.api.llm_api.LLMAPI.create", return_value=fake):
        status = run_finalize_pass(args, artifacts)
    check("三次失败 → 返回 1", status == 1 and fake.calls == 3)

with tempfile.TemporaryDirectory() as td:
    args, artifacts = make_artifacts(Path(td))
    artifacts["event_path"].write_text('{"type": "thread.started", "thread_id": "t1"}\n')
    fake = FakeAPI(["不应被调用"])
    with mock.patch("sr_agent.api.llm_api.LLMAPI.create", return_value=fake):
        status = run_finalize_pass(args, artifacts)
    check("空日志直接失败且不调 API", status == 1 and fake.calls == 0)

print(f"\n全部通过（{PASS} 项断言）")
