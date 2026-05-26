#!/usr/bin/env python
"""Collect and compare property tool experiment results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_run_sr_agent_result(exp_name):
    """Load result from run_sr_agent.py experiment."""
    result_file = ROOT / "logs" / "run_sr_agent" / exp_name / "result.jsonl"
    if not result_file.exists():
        return None
    results = []
    with open(result_file) as f:
        for line in f:
            results.append(json.loads(line))
    if not results:
        return None
    return min(results, key=lambda r: r.get("best_mse", float("inf")))


def load_bench_result(exp_name, problem_name):
    """Load result from bench_sr_agent.py experiment."""
    result_file = ROOT / "logs" / "bench_sr_agent" / exp_name / "results" / f"lsrtransform_{problem_name}.jsonl"
    if not result_file.exists():
        return None
    results = []
    with open(result_file) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)
    if not results:
        return None
    return min(results, key=lambda r: r.get("best_mse", float("inf")))


def load_tool_calls(exp_name, is_bench=False, problem_name=None):
    """Load tool call sequences."""
    if is_bench:
        exp_dir = ROOT / "logs" / "bench_sr_agent" / exp_name / "experiments"
        if problem_name:
            matching = list(exp_dir.glob(f"{problem_name}_*"))
            if matching:
                tc_file = matching[-1] / "tool_calls.jsonl"
            else:
                return []
        else:
            return []
    else:
        tc_file = ROOT / "logs" / "run_sr_agent" / exp_name / "tool_calls.jsonl"

    if not tc_file.exists():
        return []
    calls = []
    with open(tc_file) as f:
        for line in f:
            calls.append(json.loads(line))
    return calls


def extract_tool_sequence(calls):
    """Extract tool call name sequence."""
    return [c.get("name", "unknown") for c in calls]


def extract_metrics(result, is_run=False):
    """Extract key metrics from a result dict."""
    if result is None:
        return {"formula": "N/A", "mse": float("nan"), "time": "N/A",
                "tokens": "N/A", "cost": "N/A", "n_calls": "N/A", "tools_usage": "N/A"}

    if is_run:
        return {
            "formula": result.get("best_formula", "N/A"),
            "mse": result.get("best_mse", float("nan")),
            "r2": result.get("best_r2", float("nan")),
            "time": result.get("duration_seconds", "N/A"),
            "tokens": result.get("token_usage", "N/A"),
            "cost": result.get("money_usage", "N/A"),
            "n_calls": result.get("tools_usage", "N/A"),
        }
    else:
        return {
            "formula": result.get("best_formula", "N/A"),
            "mse": result.get("best_mse", float("nan")),
            "r2": result.get("best_r2", float("nan")),
            "time": result.get("duration_seconds", result.get("total_time", "N/A")),
            "tokens": result.get("token_usage", "N/A"),
            "cost": result.get("money_usage", "N/A"),
            "n_calls": result.get("tools_usage", "N/A"),
        }


def print_results():
    formulas = [
        ("A", "sin(x1-x2)", "run", "prop_tool_compare_baseline_sinx1x2", "prop_tool_compare_with_prop_sinx1x2", None),
        ("B", "II.24.17_0_1", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "II.24.17_0_1"),
        ("C", "II.13.23_1_0", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "II.13.23_1_0"),
        ("D", "I.34.14_2_0", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "I.34.14_2_0"),
        ("E", "I.30.3_0_0", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "I.30.3_0_0"),
        ("F", "I.50.26_3_0", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "I.50.26_3_0"),
        ("G", "III.15.12_0_0", "bench", "prop_tool_compare_baseline", "prop_tool_compare_with_prop", "III.15.12_0_0"),
    ]

    all_results = {}

    for label, name, mode, baseline_exp, prop_exp, problem in formulas:
        is_run = mode == "run"

        if is_run:
            r_base = load_run_sr_agent_result(baseline_exp)
            r_prop = load_run_sr_agent_result(prop_exp)
            tc_base = load_tool_calls(baseline_exp, is_bench=False)
            tc_prop = load_tool_calls(prop_exp, is_bench=False)
        else:
            r_base = load_bench_result(baseline_exp, problem)
            r_prop = load_bench_result(prop_exp, problem)
            tc_base = load_tool_calls(baseline_exp, is_bench=True, problem_name=problem)
            tc_prop = load_tool_calls(prop_exp, is_bench=True, problem_name=problem)

        m_base = extract_metrics(r_base, is_run)
        m_prop = extract_metrics(r_prop, is_run)

        seq_base = extract_tool_sequence(tc_base)
        seq_prop = extract_tool_sequence(tc_prop)

        prop_tool_results = [c for c in tc_prop if c.get("name") == "predict_property"]

        all_results[label] = {
            "name": name,
            "baseline": m_base,
            "with_prop": m_prop,
            "baseline_tool_sequence": seq_base,
            "prop_tool_sequence": seq_prop,
            "prop_tool_calls": prop_tool_results,
        }

        print(f"\n{'='*80}")
        print(f"Formula {label}: {name}")
        print(f"{'='*80}")

        for cfg_name, metrics in [("Baseline", m_base), ("With Prop", m_prop)]:
            print(f"\n  {cfg_name}:")
            print(f"    Best formula: {metrics['formula']}")
            mse = metrics['mse']
            mse_str = f"{mse:.2e}" if isinstance(mse, (int, float)) else str(mse)
            print(f"    MSE: {mse_str}")
            print(f"    Time: {metrics['time']}")
            print(f"    Tokens: {metrics['tokens']}")
            print(f"    Cost: {metrics['cost']}")
            print(f"    Tools: {metrics['n_calls']}")

        if seq_base:
            print(f"\n  Baseline tool sequence ({len(seq_base)} calls):")
            print(f"    {' -> '.join(seq_base[:20])}")
            if len(seq_base) > 20:
                print(f"    ... ({len(seq_base)} total)")
        if seq_prop:
            print(f"\n  With Prop tool sequence ({len(seq_prop)} calls):")
            print(f"    {' -> '.join(seq_prop[:20])}")
            if len(seq_prop) > 20:
                print(f"    ... ({len(seq_prop)} total)")

        if prop_tool_results:
            print(f"\n  Property prediction outputs ({len(prop_tool_results)} calls):")
            for i, ptc in enumerate(prop_tool_results[:2]):
                print(f"    Call {i+1}: {ptc.get('result_str', 'N/A')[:200]}")

    output_file = ROOT / "logs" / "exp_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str, allow_nan=True)
    print(f"\n\nFull results saved to {output_file}")


if __name__ == "__main__":
    print_results()
