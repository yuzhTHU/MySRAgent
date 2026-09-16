"""Compare nd2py BFGSFit with SRAgent's structure-aware constant optimizer.

The benchmark deterministically selects the three numeric-only LSR-Synth
formulas with the most fitable constants from biology and materials science.
Every method receives a copy of the same randomly initialized expression.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import datasets
import h5py
import nd2py as nd
import numpy as np

from sr_agent.utils.constant_optimizer import fit_constants


DOMAINS = ("bio_pop_growth", "matsci")


def normalize_expression(expression: str, symbols: list[str]) -> str:
    expression = expression.replace("^", "**").replace("Abs(", "abs(")
    for symbol in symbols:
        expression = re.sub(rf"\b{re.escape(symbol)}\b\(t\)", symbol, expression)
    return expression


def load_cases(data_root: Path, per_domain: int) -> list[dict]:
    candidates = []
    for domain in DOMAINS:
        parquet = next((data_root / "data").glob(f"lsr_synth_{domain}-*.parquet"))
        dataset = datasets.load_dataset("parquet", data_files=str(parquet), split="train")
        domain_candidates = []
        for row in dataset:
            expression = normalize_expression(row["expression"], row["symbols"])
            try:
                parsed = nd.parse(expression)
            except Exception:
                continue
            variables = {
                node.name for node in parsed.iter_preorder() if isinstance(node, nd.Variable)
            }
            if not variables <= set(row["symbols"]):
                continue
            parameter_count = len(parsed.get_numbers(fitable_only=True))
            domain_candidates.append((parameter_count, row["name"], expression, row["symbols"]))
        for parameter_count, name, expression, symbols in sorted(domain_candidates, reverse=True)[:per_domain]:
            candidates.append({
                "domain": domain,
                "name": name,
                "expression": expression,
                "symbols": symbols,
                "parameter_count": parameter_count,
            })

    with h5py.File(data_root / "lsr_bench_data.hdf5", "r") as handle:
        for case in candidates:
            samples = handle[f"/lsr_synth/{case['domain']}/{case['name']}/train"][...].astype(float)
            case["data"] = {
                symbol: samples[:, index] for index, symbol in enumerate(case["symbols"])
            }
    return candidates


def nmse(expression: nd.Symbol, data: dict[str, np.ndarray], target: np.ndarray) -> float:
    try:
        prediction = np.asarray(expression.eval(data), dtype=float).reshape(-1)
    except Exception:
        return float("inf")
    if prediction.size != target.size or not np.all(np.isfinite(prediction)):
        return float("inf")
    variance = float(np.var(target))
    return float(np.mean((prediction - target) ** 2) / variance) if variance > 0 else float("inf")


def summarize(values: list[dict], method: str) -> dict:
    errors = np.asarray([item["nmse"] for item in values], dtype=float)
    times = np.asarray([item["seconds"] for item in values], dtype=float)
    return {
        "runs": len(values),
        "success_nmse_lt_1e-10": int(np.count_nonzero(errors < 1e-10)),
        "usable_nmse_lt_1e-6": int(np.count_nonzero(errors < 1e-6)),
        "median_nmse": float(np.median(errors)),
        "median_seconds": float(np.median(times)),
        "mean_seconds": float(np.mean(times)),
        "method": method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/llm-srbench-data"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--per-domain", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.data_root, args.per_domain)
    results = []
    for case in cases:
        target_name = case["symbols"][0]
        target = case["data"][target_name]
        features = {name: case["data"][name] for name in case["symbols"][1:]}
        for seed in range(args.seeds):
            initial = nd.parse(case["expression"])
            rng = np.random.default_rng(seed)
            for parameter in initial.get_numbers(fitable_only=True):
                parameter.value = float(rng.uniform(-1.0, 1.0))

            bfgs_expression = initial.copy()
            start = time.perf_counter()
            nd.BFGSFit(bfgs_expression).fit(features, target)
            bfgs_seconds = time.perf_counter() - start
            results.append({
                "name": case["name"], "domain": case["domain"], "seed": seed,
                "parameter_count": case["parameter_count"], "method": "nd.BFGSFit",
                "nmse": nmse(bfgs_expression, features, target), "seconds": bfgs_seconds,
            })

            start = time.perf_counter()
            structure_expression = fit_constants(initial, features, target)
            structure_seconds = time.perf_counter() - start
            results.append({
                "name": case["name"], "domain": case["domain"], "seed": seed,
                "parameter_count": case["parameter_count"], "method": "structure-aware",
                "nmse": nmse(structure_expression, features, target),
                "seconds": structure_seconds,
            })
            print(
                f"{case['name']} seed={seed}: "
                f"BFGS nmse={results[-2]['nmse']:.3e} time={bfgs_seconds:.3f}s; "
                f"structure nmse={results[-1]['nmse']:.3e} time={structure_seconds:.3f}s",
                flush=True,
            )

    methods = ("nd.BFGSFit", "structure-aware")
    summary = {
        method: summarize([item for item in results if item["method"] == method], method)
        for method in methods
    }
    per_formula = {}
    for case in cases:
        per_formula[case["name"]] = {
            method: summarize([
                item for item in results
                if item["name"] == case["name"] and item["method"] == method
            ], method)
            for method in methods
        }
    report = {
        "selection": "top parameter-count numeric-only formulas per domain",
        "initialization": "all fitable constants iid Uniform(-1, 1), paired by seed",
        "success_definition": "training NMSE < 1e-10",
        "cases": [{key: value for key, value in case.items() if key != "data"} for case in cases],
        "summary": summary,
        "per_formula": per_formula,
        "runs": results,
    }
    rendered = json.dumps(report, indent=2, allow_nan=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
