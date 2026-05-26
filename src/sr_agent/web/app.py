from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import nd2py as nd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
DEFAULT_LOG_DIR = Path.cwd() / "logs"


def create_app(log_dir: str | Path = DEFAULT_LOG_DIR) -> FastAPI:
    app = FastAPI(title="SR Agent Search Viewer")
    app.state.log_dir = Path(log_dir).resolve()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/runs")
    def list_runs():
        runs = []
        for run_dir in _iter_run_dirs(app.state.log_dir):
            manifest = _read_json(run_dir / "manifest.json") or {}
            record_count, last_seq = _record_stats(run_dir / "records.jsonl")
            runs.append({
                "run_id": run_dir.name,
                "path": str(run_dir),
                "manifest": manifest,
                "record_count": record_count,
                "last_seq": last_seq,
                "mtime": (run_dir / "records.jsonl").stat().st_mtime,
            })
        runs.sort(key=lambda item: item["mtime"], reverse=True)
        return {"runs": runs}

    @app.get("/api/runs/{run_id}/records")
    def list_records(run_id: str, after_seq: int = Query(0, ge=0), include_detail: bool = False):
        run_dir = _resolve_run_dir(app.state.log_dir, run_id)
        all_records = list(_read_records(run_dir / "records.jsonl"))
        records_by_id = {record.get("node_id"): record for record in all_records if record.get("node_id")}
        records = []
        for record in all_records:
            if int(record.get("seq", 0)) <= after_seq:
                continue
            record = _with_core_derivatives(record, records_by_id)
            records.append(record if include_detail else _strip_detail(record))
        return {"records": records}

    @app.get("/api/runs/{run_id}/records/{node_id}")
    def get_record(run_id: str, node_id: str):
        run_dir = _resolve_run_dir(app.state.log_dir, run_id)
        all_records = list(_read_records(run_dir / "records.jsonl"))
        records_by_id = {record.get("node_id"): record for record in all_records if record.get("node_id")}
        for record in all_records:
            if record.get("node_id") == node_id:
                return _with_core_derivatives(record, records_by_id)
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    @app.get("/api/runs/{run_id}/stream")
    async def stream_records(run_id: str, after_seq: int = Query(0, ge=0)):
        run_dir = _resolve_run_dir(app.state.log_dir, run_id)
        records_path = run_dir / "records.jsonl"

        async def event_source():
            next_seq = after_seq
            while True:
                batch = []
                all_records = list(_read_records(records_path))
                records_by_id = {record.get("node_id"): record for record in all_records if record.get("node_id")}
                for record in all_records:
                    seq = int(record.get("seq", 0))
                    if seq > next_seq:
                        batch.append(_strip_detail(_with_core_derivatives(record, records_by_id)))
                        next_seq = max(next_seq, seq)
                if batch:
                    yield f"data: {json.dumps({'records': batch}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return app


app = create_app()


def _iter_run_dirs(log_dir: Path):
    if not log_dir.exists():
        return
    for path in log_dir.rglob("records.jsonl"):
        run_dir = path.parent
        if (run_dir / "manifest.json").exists():
            yield run_dir


def _resolve_run_dir(log_dir: Path, run_id: str) -> Path:
    matches = [path for path in _iter_run_dirs(log_dir) if path.name == run_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    matches.sort(key=lambda path: (path / "records.jsonl").stat().st_mtime, reverse=True)
    return matches[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _read_records(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def _record_stats(path: Path) -> tuple[int, int]:
    count = 0
    last_seq = 0
    for record in _read_records(path):
        count += 1
        last_seq = max(last_seq, int(record.get("seq", 0)))
    return count, last_seq


def _strip_detail(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "detail"}


def _with_core_derivatives(
    record: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = _with_formula_latex(record)
    return _with_tool_summary(record, records_by_id or {})


def _with_formula_latex(record: dict[str, Any]) -> dict[str, Any]:
    core = record.get("core")
    if not isinstance(core, dict) or core.get("formula_latex"):
        return record
    formula = core.get("formula")
    if not formula:
        return record
    record = dict(record)
    core = dict(core)
    try:
        core["formula_latex"] = nd.parse(str(formula).replace("^", "**")).to_str(latex=True)
    except Exception:
        core["formula_latex"] = None
    record["core"] = core
    return record


def _with_tool_summary(record: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    core = record.get("core")
    if not isinstance(core, dict) or core.get("tool_summary"):
        return record
    record = dict(record)
    core = dict(core)
    output_counts: dict[str, int] = {}
    for name in core.get("tool_names") or []:
        if name:
            output_counts[str(name)] = output_counts.get(str(name), 0) + 1
    prompt_counts = _prompt_tool_counts(record.get("detail", {}).get("prompt", []))
    ancestor_counts = _ancestor_tool_counts(record, records_by_id)
    for name, count in ancestor_counts.items():
        prompt_counts[name] = max(prompt_counts.get(name, 0), count)
    names = sorted(set(prompt_counts) | set(output_counts))
    core["tool_summary"] = [
        {
            "name": name,
            "display_name": name.replace("_", " ").title(),
            "prompt_count": prompt_counts.get(name, 0),
            "output_count": output_counts.get(name, 0),
        }
        for name in names
    ]
    record["core"] = core
    return record


def _ancestor_tool_counts(
    record: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    seen: set[str] | None = None,
) -> dict[str, int]:
    seen = seen or set()
    counts: dict[str, int] = {}
    for parent in record.get("parents") or []:
        if not isinstance(parent, dict):
            continue
        parent_id = parent.get("node_id")
        if not parent_id or parent_id in seen:
            continue
        parent_record = records_by_id.get(parent_id)
        if not parent_record:
            continue
        seen.add(parent_id)
        for name in (parent_record.get("core") or {}).get("tool_names") or []:
            if name:
                name = str(name)
                counts[name] = counts.get(name, 0) + 1
        for name, count in _ancestor_tool_counts(parent_record, records_by_id, seen).items():
            counts[name] = counts.get(name, 0) + count
    return counts


def _prompt_tool_counts(prompt: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(prompt, list):
        return counts
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and msg.get("name"):
            name = str(msg["name"])
            counts[name] = counts.get(name, 0) + 1
    if counts:
        return counts
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or (call.get("function") or {}).get("name")
            if name:
                name = str(name)
                counts[name] = counts.get(name, 0) + 1
    return counts
