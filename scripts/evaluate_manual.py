# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""对 status=manual 的题跑 judge，评估 manual 补充（ChatGPT 提取）的性能。

judge 流程与 bench 完全一致（get_symbolic_acc：数值比对 → nsimplify → LLM judge）。
ground-truth 取自 results/*.jsonl 中 error 行记录的 gt_expression（与匿名化后
的变量名一致），数据取自每题实验目录的 context.npz。

用法:
    venv/bin/python scripts/evaluate_manual.py <experiments_dir> [--no-llm-judge]
    venv/bin/python scripts/evaluate_manual.py <experiments_dir> --llm-provider openrouter --llm-model qwen/qwen3.5-flash-02-23

前置: LLM judge 需要 source ~/.bashrc && use_openrouter
只评估 status=manual 的题。
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nd2py as nd  # noqa: E402
from sr_agent.utils.logger import _setup_custom_levels  # noqa: E402
from sr_agent.utils.symbolic_acc import get_symbolic_acc  # noqa: E402

_setup_custom_levels()  # judge 内部使用 TRACE 日志级别，需先注册


def parse_expr(expression: str) -> nd.Symbol:
    """与 bench run() 终检相同的预处理后解析。"""
    return nd.parse(expression.strip().replace("^", "**").replace("np.", "").replace("math.", ""))


def find_gt_expression(results_dir: Path, equation_id: str) -> str | None:
    for result_file in sorted(results_dir.glob(f"*_{equation_id}.jsonl")):
        try:
            for line in result_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "gt_expression" in row:
                    return str(row["gt_expression"])
        except OSError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="对 status=manual 的题跑 judge")
    parser.add_argument("experiments", help="experiments 目录")
    parser.add_argument("--no-llm-judge", action="store_true", help="跳过 LLM judge（纯数值+nsimplify，更快不花钱）")
    parser.add_argument("--llm-provider", default="openrouter")
    parser.add_argument("--llm-model", default="qwen/qwen3.5-flash-02-23")
    args = parser.parse_args()

    experiments = Path(args.experiments)
    if not experiments.is_dir():
        print(f"目录不存在: {experiments}")
        return 1
    results_dir = experiments.parent / "results"

    manual = []
    for result_path in sorted(experiments.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "manual" and result.get("discovered_expression"):
            manual.append((result_path.parent, result))
    print(f"发现 {len(manual)} 个 manual 题")

    equiv = not_equiv = unparsable = 0
    for problem_dir, result in manual:
        name = problem_dir.name
        equation_id = result.get("equation_id", "")
        expression = str(result.get("discovered_expression") or "")
        try:
            f_pred = parse_expr(expression)
        except Exception as e:
            print(f"[无法解析] {name}: {expression[:50]} ({type(e).__name__})")
            unparsable += 1
            continue
        gt_str = find_gt_expression(results_dir, equation_id)
        if not gt_str:
            print(f"[缺 GT] {name}: results 里找不到 {equation_id} 的 gt_expression，跳过")
            continue
        try:
            f_true = parse_expr(gt_str)
        except Exception as e:
            print(f"[GT 解析失败] {name}: {gt_str[:50]} ({type(e).__name__})，跳过")
            continue
        try:
            data = np.load(problem_dir / "context.npz", allow_pickle=True)["data"].item()
        except Exception as e:
            print(f"[缺数据] {name}: context.npz 读取失败 ({type(e).__name__})，跳过")
            continue
        try:
            verdict = get_symbolic_acc(
                f_true, f_pred, data,
                llm_judge=not args.no_llm_judge,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                wait_for_human=False,
            )
        except Exception as e:
            print(f"[judge 异常] {name}: {type(e).__name__}: {e}")
            not_equiv += 1
            continue
        if verdict:
            equiv += 1
            print(f"[等价] {name}")
        else:
            not_equiv += 1
            print(f"[不等价] {name}: {expression[:60]}")

    total = equiv + not_equiv
    print(f"\n完成: 等价 {equiv} / 不等价 {not_equiv} / 无法解析 {unparsable}")
    if total:
        print(f"manual 补充成功率: {equiv}/{total} = {equiv / total:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
