# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import os
import json
import requests
import argparse
import logging
import pandas as pd
from socket import gethostname
from sr_agent.utils import tag2ansi

_logger = logging.getLogger("sr_agent.bench_server.client")


def results_to_records(results_df: pd.DataFrame) -> list[dict]:
    records_df = results_df.reset_index()
    if "level_1" in records_df.columns:
        records_df = records_df.rename(columns={"level_1": "metric"})
    if "index" in records_df.columns:
        records_df = records_df.rename(columns={"index": "metric"})
    return json.loads(records_df.to_json(orient="records"))


def submit_benchmark(args: argparse.Namespace, results_df: pd.DataFrame) -> str:
    if not args.bench_server_url:
        return ""

    base_url = args.bench_server_url.rstrip("/")
    payload = {
        "exp_name": args.exp_name,
        "host": gethostname(),
        "command": args.command,
        "submitted_by": os.environ.get("USER") or os.environ.get("USERNAME"),
        "metadata": {
            "data_dir": args.data_dir,
            "n_nodes": args.n_nodes,
            "seed": args.seed,
            "stimulus": args.stimulus,
            "sampling_hz": args.sampling_hz,
            "normalize": args.normalize,
            "algorithm": args.algorithm,
            "hist_steps": args.hist_steps,
            "max_rollout_steps": args.max_rollout_steps,
        },
        "results": results_to_records(results_df),
    }
    try:
        response = requests.post(
            f"{base_url}/submit",
            json=payload,
            timeout=args.bench_timeout,
        )
        response.raise_for_status()
        response = requests.get(
            f"{base_url}/leaderboard",
            params={"limit": args.bench_leaderboard_limit},
            timeout=args.bench_timeout,
        )
        response.raise_for_status()
        leaderboard = response.json().get("leaderboard", [])
    except Exception as exc:
        _logger.warning(f"Failed to submit benchmark results to {base_url}: {exc}")
        return ""

    return format_leaderboard(leaderboard)


def format_leaderboard(leaderboard: list[dict]) -> str:
    if not leaderboard:
        return tag2ansi("[yellow]Leaderboard is empty.[reset]")

    rows = []
    for i, entry in enumerate(leaderboard, start=1):
        score = entry.get("score")
        score_text = "nan" if score is None else f"{score:.4g}"
        rows.append({
            "rank": i,
            "score": score_text,
            "exp_name": entry.get("exp_name", ""),
            "algorithm": entry.get("algorithm", ""),
            "metric": entry.get("metric", ""),
            "split": entry.get("split", ""),
            "submitted_at": entry.get("submitted_at", ""),
        })
    leaderboard_df = pd.DataFrame(rows)
    return tag2ansi(
        "[cyan bold]Benchmark leaderboard[reset]\n"
        "[gray]" + "=" * 60 + "[reset]\n"
        f"{leaderboard_df.to_string(index=False)}\n"
        "[gray]" + "=" * 60 + "[reset]"
    )
