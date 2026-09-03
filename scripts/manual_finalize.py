# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""manual finalize：配合 ChatGPT（网页版/桌面版）手动补充 missing 题的公式。

学长方案：对 resume 也救不回的 missing 题，把主 pass 的探索日志交给好模型
（ChatGPT），让它输出 result.json 内容；解析成功以 status="manual" 写回
该题 result.json，与 agent 自提交（completed）/resume 补交（resume）区分。

工作流（中间对话由你手动完成）：
  1. export：为每道 missing 题生成一个干净 txt（完整指令 + 变量名 + 整理后
     的日志），放在 manual/prompts/<题目录名>.txt
  2. 把 txt 拖进 ChatGPT（桌面版直接拖文件，网页版复制粘贴内容），
     让它按文件要求输出 JSON
  3. 把 ChatGPT 回复的 JSON 存为 manual/replies/<题目录名>.txt
  4. import：解析 replies 里的 JSON，合法则以 status="manual" 写回 result.json

用法:
    venv/bin/python scripts/manual_finalize.py <experiments_dir> export [--limit 10]
    venv/bin/python scripts/manual_finalize.py <experiments_dir> import [--limit 10]

只处理 status=missing 的题；正在跑的题自动跳过；result.json 其余字段原样保留。
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

_MAX_LOG_BYTES = 400_000

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
    """从 ChatGPT 回复文本中解析包含 discovered_expression 字段的 JSON 对象；失败返回 None。"""
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


def collect_missing(experiments: Path, limit: int | None) -> list[Path]:
    missing = []
    for result_path in sorted(experiments.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "missing":
            missing.append(result_path.parent)
    if limit:
        missing = missing[:limit]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="manual finalize：配合 ChatGPT 手动补充 missing 题")
    parser.add_argument("experiments", help="experiments 目录，如 logs/.../codex_flash_no-tool_anonymize_v10/experiments")
    parser.add_argument("action", choices=["export", "import", "status"], help="export: 生成给 ChatGPT 的 txt; import: 解析 ChatGPT 回复并写回 result.json; status: 查看各状态题数")
    parser.add_argument("--limit", type=int, default=None, help="最多处理的题数")
    parser.add_argument("--save-dir", default=None, help="export 用：GPT 桌面版所在电脑的目录（prompt 会让 GPT 把 JSON 存到该目录），如 'C:/Users/xxx/Desktop/replies'")
    args = parser.parse_args()

    experiments = Path(args.experiments)
    if not experiments.is_dir():
        print(f"目录不存在: {experiments}")
        return 1
    prompts_dir = experiments / "manual" / "prompts"
    replies_dir = experiments / "manual" / "replies"

    if args.action == "status":
        cnt: dict[str, int] = {}
        missing_list = []
        for result_path in sorted(experiments.glob("*/result.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            st = result.get("status", "unknown")
            cnt[st] = cnt.get(st, 0) + 1
            if st == "missing":
                missing_list.append(result_path.parent.name)
        print(f"状态分布: {dict(cnt)}")
        print(f"待补充 missing（{len(missing_list)} 个）:")
        for name in missing_list:
            print(f"  {name}")
        return 0

    missing = collect_missing(experiments, args.limit)
    print(f"发现 {len(missing)} 个 missing 题" + (f"（--limit {args.limit}）" if args.limit else ""))

    if args.action == "export":
        prompts_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for problem_dir in missing:
            log_text = extract_log_text(problem_dir / "codex_events.jsonl")
            if not log_text.strip():
                print(f"[跳过] {problem_dir.name}: 无探索记录")
                continue
            features = features_from_manifest(problem_dir)
            feature_hint = f"The feature variables are: {', '.join(features)}. " if features else ""
            save_instruction = ""
            if args.save_dir:
                save_instruction = (
                    f'\n\nAfter outputting the JSON, also save it as a file named '
                    f'"{problem_dir.name}.json" in the directory "{args.save_dir}" on this computer. '
                    f"If the directory does not exist, create it first."
                )
            prompt = (
                _SYSTEM_PROMPT + "\n\n" +
                feature_hint + "Exploration log:\n\n" + log_text +
                "\n\nNow output the result JSON." + save_instruction
            )
            out = prompts_dir / f"{problem_dir.name}.txt"
            out.write_text(prompt, encoding="utf-8")
            made += 1
            print(f"[导出] {problem_dir.name} ({len(prompt.encode('utf-8')) // 1024}KB)")
        print(f"\n完成: 导出 {made} 个 prompt 到 {prompts_dir}")
        print(f"下一步: 把每个 txt 拖进 ChatGPT，把它的 JSON 回复存到 {replies_dir}/ 下")
        print(f"        （文件名与 prompt 相同：<题目录名>.txt），然后执行 import。")
        return 0

    # import
    if not replies_dir.is_dir():
        print(f"replies 目录不存在: {replies_dir}（请先 export 并完成 ChatGPT 对话）")
        return 1
    ok = invalid = skipped = 0
    for problem_dir in missing:
        reply_path = replies_dir / f"{problem_dir.name}.txt"
        if not reply_path.exists():
            reply_path = replies_dir / f"{problem_dir.name}.json"
        if not reply_path.exists():
            skipped += 1
            continue
        content = reply_path.read_text(encoding="utf-8")
        parsed = parse_finalize_json(content)
        expression = str(parsed.get("discovered_expression") or "").strip() if parsed else ""
        if not expression:
            print(f"[无效] {problem_dir.name}: 回复无法解析为含公式的 JSON，保持 missing")
            invalid += 1
            continue
        result_path = problem_dir / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        if result.get("status") != "missing":
            print(f"[跳过] {problem_dir.name}: 当前 status={result.get('status')}，只允许覆盖 missing")
            skipped += 1
            continue
        result["discovered_expression"] = expression
        result["status"] = "manual"
        result["notes"] = "manual finalize by ChatGPT"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        ok += 1
        print(f"[补充] {problem_dir.name}: {expression[:60]}")
    print(f"\n完成: 补充 {ok} 个, 无效 {invalid} 个, 跳过 {skipped} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
