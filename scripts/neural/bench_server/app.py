# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parents[3]
RESULTS_DIR = ROOT / "logs" / "neural" / "bench_server" / "results"
DEFAULT_SPLIT = "eval_other_times"
DEFAULT_METRIC = "pearson>0.5"


def sanitize_filename(value: str) -> str:
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub("_", value.strip())
    return (value or "unnamed")[:255]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_submissions(results_dir: Path) -> list[dict]:
    submissions = []
    if not results_dir.exists():
        return submissions
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                submission = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        submission["_path"] = str(path)
        submissions.append(submission)
    return submissions


def extract_score(submission: dict, split: str, metric: str) -> float | None:
    for row in submission.get("results", []):
        if row.get("split") == split and row.get("metric") == metric:
            score = row.get("seconds", row.get("step", row.get("value")))
            if isinstance(score, int | float) and math.isfinite(score):
                return float(score)
    return None


def build_leaderboard(
    results_dir: Path,
    *,
    limit: int = 10,
    split: str = DEFAULT_SPLIT,
    metric: str = DEFAULT_METRIC,
) -> list[dict]:
    entries = []
    for submission in load_submissions(results_dir):
        score = extract_score(submission, split=split, metric=metric)
        metadata = submission.get("metadata", {})
        entries.append({
            "id": submission.get("id"),
            "exp_name": submission.get("exp_name"),
            "algorithm": metadata.get("algorithm"),
            "score": score,
            "metric": metric,
            "split": split,
            "submitted_at": submission.get("submitted_at"),
            "submitted_by": submission.get("submitted_by"),
            "host": submission.get("host"),
        })
    entries.sort(
        key=lambda entry: (
            entry["score"] is not None,
            entry["score"] if entry["score"] is not None else float("-inf"),
            entry.get("submitted_at") or "",
        ),
        reverse=True,
    )
    return entries[: max(0, limit)]


class BenchHandler(BaseHTTPRequestHandler):
    server_version = "NeuralBenchServer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/leaderboard", "/leatherboard"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        query = parse_qs(parsed.query)
        limit = parse_int(query.get("limit", ["10"])[0], default=10)
        split = query.get("split", [DEFAULT_SPLIT])[0]
        metric = query.get("metric", [DEFAULT_METRIC])[0]
        self.write_json({
            "leaderboard": build_leaderboard(
                RESULTS_DIR,
                limit=limit,
                split=split,
                metric=metric,
            )
        })

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self.read_json()
            submission = normalize_submission(payload)
            filename = sanitize_filename(f"{submission['submitted_at']}_{submission['id']}.json")
            write_json_atomic(RESULTS_DIR / filename, submission)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.write_json({
            "ok": True,
            "id": submission["id"],
            "leaderboard": build_leaderboard(RESULTS_DIR),
        })

    def read_json(self) -> dict:
        length = parse_int(self.headers.get("Content-Length"), default=-1)
        if length < 0:
            raise ValueError("Missing Content-Length.")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def write_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(json_safe(data), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} {self.address_string()} {fmt % args}")


def normalize_submission(payload: dict) -> dict:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Field 'results' must be a list.")

    submitted_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_name = str(payload.get("exp_name") or "unnamed")
    host = str(payload.get("host") or "unknown-host")
    submission_id = sanitize_filename(f"{submitted_at}_{host}_{exp_name}")
    return {
        "id": submission_id,
        "submitted_at": submitted_at,
        "exp_name": exp_name,
        "host": host,
        "submitted_by": payload.get("submitted_by"),
        "command": payload.get("command"),
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "results": results,
    }


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural benchmark result server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), BenchHandler)
    print(f"Serving neural benchmark server on http://{args.host}:{args.port}")
    print(f"Saving submissions to {RESULTS_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
