# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""manual finalize：用更强的模型从 missing 题的探索日志提取公式，补充结果。

学长方案：对 resume 也救不回的 missing 题，把主 pass 的探索日志发给好模型
（默认 openrouter 的 openai/gpt-5.5），让它输出 result.json 内容；解析成功则
以 status="manual" 写回该题 result.json，与 agent 自提交（completed）和
resume 补交（resume）明确区分，供后续统一 judge 评估 missing 题的平均性能。

用法:
    venv/bin/python scripts/manual_finalize.py <experiments_dir> [--model openai/gpt-5.5] [--limit 10] [--dry-run]

前置: source ~/.bashrc && use_openrouter（OPENROUTER_API_KEY 供 API 调用）
只处理 status=missing 的题；正在跑（started）的题自动跳过。
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sr_agent.api.llm_api import LLMAPI  # noqa: E402

_MAX_LOG_BYTES = 400_000
_MAX_ATTEMPTS = 3

_SYSTEM_PROMPT = (
    "You are a formula-extraction assistant for a symbolic-regression benchmark. "
    "You will receive the exploration log of a coding agent that searched for the formula "
    "relating the input features to the target variable, but did not submit a final answer "
    "before its time budget ran out. Your only job is to identify the best candidate formula "
    "in the log and output it as JSON.\n"
    "Rules:\n"
    "- Pick the best candidate formula in the log (typically the one with the highest R² / lowest error).\n"
    "- Transcribe the formula exactly as printed, character by character. Do not modify, simplify, or re-derive it.\n"
    "- The formula may use only the feature variables listed in the user message, plus the constants pi and e.\n"
    "- Output only one JSON object and nothing else, with exactly these fields:\n"
    '  {"discovered_expression": "<formula>", "status": "completed", "notes": "<short note>"}\n'
    "- If you cannot identify any credible formula, output "
    '{"discovered_expression": null, "status": "failed", "notes": "no credible formula in log"}.\n'
    "- Do not wrap the JSON in code fences and do not add any other text."
)


def extract_log_text(event_path: Path, max_bytes: int = _MAX_LOG_BYTES) -> str:
    """从主 pass 事件流提取 (命令, 输出) 文本块，超限按 头25%+尾75% 字节预算截断。"""
    pairs: list[tuple[str, str]] = []
    pending: dict[str, str] = {}
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
                item = event.get("item") or {}
                if event.get("type") == "item.started" and item.get("type") == "command_execution":
                    raw_command = item.get("command")
                    command = raw_command if isinstance(raw_command, str) else " ".join(raw_command or [])
                    pending[item.get("id", "")] = command
                elif event.get("type") == "item.completed" and item.get("type") == "command_execution":
                    command = pending.pop(item.get("id", ""), "")
                    output = item.get("aggregated_output") or ""
                    if command or output:
                        pairs.append((command, output))
    except OSError:
        pass

    def render(selected: list[tuple[str, str]]) -> str:
        blocks = []
        for i, (command, output) in enumerate(selected):
            blocks.append(f"[Command {i + 1}]\n{command}\n[Output {i + 1}]\n{output}")
        return "\n\n".join(blocks)

    text = render(pairs)
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    head_budget = int(max_bytes * 0.25)
    tail_budget = max_bytes - head_budget
    head: list[tuple[str, str]] = []
    acc = 0
    for pair in pairs:
        size = len(pair[0].encode("utf-8")) + len(pair[1].encode("utf-8"))
        if acc + size > head_budget:
            break
        head.append(pair)
        acc += size
    tail: list[tuple[str, str]] = []
    acc = 0
    for pair in reversed(pairs):
        size = len(pair[0].encode("utf-8")) + len(pair[1].encode("utf-8"))
        if acc + size > tail_budget:
            break
        tail.append(pair)
        acc += size
    tail.reverse()
    return render(head + tail)


def parse_finalize_json(content: str) -> dict | None:
    """从 LLM 回复文本中解析包含 discovered_expression 字段的 JSON 对象；失败返回 None。"""
    import re
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "discovered_expression" not in parsed:
        return None
    return parsed


def features_from_manifest(problem_dir: Path) -> list[str]:
    try:
        manifest = json.loads((problem_dir / "manifest.json").read_text(encoding="utf-8"))
        symbols = [str(s) for s in manifest.get("symbols", [])]
        return symbols[1:]  # 首个符号是 target
    except (OSError, json.JSONDecodeError):
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", help="experiments 目录，如 logs/.../codex_flash_no-tool_anonymize_v10/experiments")
    parser.add_argument("--model", default="openai/gpt-5.5", help="openrouter 模型 ID（默认 openai/gpt-5.5）")
    parser.add_argument("--limit", type=int, default=None, help="最多补充的题数（默认全部 missing）")
    parser.add_argument("--dry-run", action="store_true", help="只预览将处理哪些题，不调用 API")
    args = parser.parse_args()

    experiments = Path(args.experiments)
    if not experiments.is_dir():
        print(f"目录不存在: {experiments}")
        return 1

    missing_dirs = []
    for result_path in sorted(experiments.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "missing":
            missing_dirs.append(result_path.parent)
    print(f"发现 {len(missing_dirs)} 个 missing 题" + (f"（--limit {args.limit}）" if args.limit else ""))
    if args.limit:
        missing_dirs = missing_dirs[:args.limit]
    if args.dry_run:
        for d in missing_dirs:
            print(f"  [dry-run] {d.name}")
        return 0

    api = LLMAPI.create(llm_provider="openrouter", llm_model=args.model)
    ok = refuse = failed = 0
    for problem_dir in missing_dirs:
        name = problem_dir.name
        log_text = extract_log_text(problem_dir / "codex_events.jsonl")
        if not log_text.strip():
            print(f"[跳过] {name}: 无探索记录")
            failed += 1
            continue
        features = features_from_manifest(problem_dir)
        feature_hint = f"The feature variables are: {', '.join(features)}. " if features else ""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": feature_hint + "Exploration log:\n\n" + log_text + "\n\nNow output the result JSON."},
        ]
        expression = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            content = ""
            try:
                for content, _, _ in api(messages, n=1, max_tokens=2048, temperature=0.0):
                    pass
            except Exception as e:
                print(f"[警告] {name}: LLM 调用失败 [{type(e).__name__}] {e}")
                continue
            parsed = parse_finalize_json(content)
            if parsed is not None:
                expression = str(parsed.get("discovered_expression") or "").strip()
                if not expression:
                    break  # 明确拒绝：日志中没有可信公式
                break
            if attempt < _MAX_ATTEMPTS:
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "Your previous reply could not be parsed as a JSON object containing a "
                    'non-null "discovered_expression" field. Reply with ONLY the JSON object.'
                )})
        if not expression:
            print(f"[拒绝/失败] {name}: 未提取到公式，保持 missing")
            refuse += 1
            continue
        # 以 status="manual" 写回，保留其余字段
        result_path = problem_dir / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        result["discovered_expression"] = expression
        result["status"] = "manual"
        result["notes"] = f"manual finalize by {args.model}"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        ok += 1
        print(f"[补充] {name}: {expression[:60]}")
    print(f"\n完成: 补充 {ok} 个, 拒绝/失败 {refuse} 个, 无记录 {failed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
