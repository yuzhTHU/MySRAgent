# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""In-memory state for one symbolic-regression search run."""
from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .tool import ToolCall, ToolCallResult

ParentRelation = Literal["continuation", "restart_seed", "context_merge"]


def json_value(value: Any) -> Any:
    """Convert runtime values to standards-compliant JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ToolCall):
        return {
            "name": value.name,
            "params": json_value(value.params),
            "id": value.id,
            "raw": json_value(value.raw),
            "raw_str": value.raw_str,
        }
    if isinstance(value, ToolCallResult):
        return {
            "ok": value.ok,
            "result": json_value(value.result),
            "result_str": value.result_str,
            "meta_data": json_value(value.meta_data),
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True, slots=True)
class SearchCoordinate:
    R: int
    C: int
    L: int
    K: int


@dataclass(frozen=True, slots=True)
class ParentLink:
    parent_node_id: str
    relation: ParentRelation

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.parent_node_id, "relation": self.relation}


@dataclass(slots=True)
class SearchNode:
    run_id: str
    node_id: str
    node_label: str
    coordinate: SearchCoordinate
    parents: tuple[ParentLink, ...]
    created_at: str
    core: dict[str, Any]
    detail: dict[str, Any]

    def to_dict(self, *, include_detail: bool = True) -> dict[str, Any]:
        record = {
            "run_id": self.run_id,
            "node_id": self.node_id,
            "node_label": self.node_label,
            "created_at": self.created_at,
            "coord": asdict(self.coordinate),
            "progress": (
                f"(R={self.coordinate.R}) x (C={self.coordinate.C}) x "
                f"(L={self.coordinate.L}) x (K={self.coordinate.K})"
            ),
            "parents": [parent.to_dict() for parent in self.parents],
            "core": json_value(self.core),
        }
        if include_detail:
            record["detail"] = json_value(self.detail)
        return record


@dataclass(slots=True)
class CandidateRecord:
    formula: str
    node_id: str
    details: dict[str, Any] = field(default_factory=dict)

    def split_metrics(self, split: str) -> dict[str, Any]:
        return (
            self.details.get("data_split_results", {})
            .get(split, {})
            .get("metrics", {})
        )

    def metric(self, name: str, split: str) -> Any:
        return self.split_metrics(split).get(name)

    @property
    def complexity(self) -> Any:
        return self.metric("complexity", "train")

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "node_id": self.node_id,
            "details": json_value(self.details),
        }

    def display_dict(self) -> dict[str, Any]:
        """Return a flattened view for UI rendering without mutating the record."""
        split = "validation" if self.split_metrics("validation") else "train"
        return self.to_dict() | {"split": split} | json_value(self.split_metrics(split))


@dataclass(slots=True)
class SearchResult:
    status: Literal["completed", "early_stopped", "interrupted", "failed"]
    progress: str
    candidates: list[CandidateRecord]
    pareto_front: list[int]
    best_candidate: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "pareto_front": list(self.pareto_front),
            "best_candidate": self.best_candidate,
        }


class SearchRunState:
    """Authoritative in-memory state for one run, with optional persistence."""

    def __init__(
        self,
        save_path: str | Path | None,
        ranking_metric: str,
        larger_is_better: bool,
        agent_metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex
        self.ranking_metric = ranking_metric
        self.larger_is_better = larger_is_better
        self.nodes: dict[str, SearchNode] = {}
        self._candidates: list[CandidateRecord] = []
        self._lock = threading.RLock()
        self._save_path = Path(save_path) if save_path is not None else None
        self._nodes_path = self._save_path / "nodes.jsonl" if self._save_path else None
        self._result_path = self._save_path / "result.json" if self._save_path else None
        if self._save_path is not None:
            self._save_path.mkdir(parents=True, exist_ok=True)
            metadata = {
                "run_id": self.run_id,
                "created_at": self.now(),
                "agent": agent_metadata or {},
            }
            self._write_json(self._save_path / "run.json", metadata)

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def node_label(R: int, C: int, L: int, K: int) -> str:
        return f"R{R}-C{C}-L{L}-K{K}"

    def node_id(self, R: int, C: int, L: int, K: int) -> str:
        return f"{self.run_id}:{self.node_label(R=R, C=C, L=L, K=K)}"

    @staticmethod
    def parent_link(parent_node_id: str, relation: ParentRelation) -> ParentLink:
        return ParentLink(parent_node_id=parent_node_id, relation=relation)

    def register_iteration(
        self,
        response_list: list,
        results_list: list,
        parents: tuple[ParentLink, ...],
        prompt: list[dict[str, Any]],
        usage: dict[str, Any],
        R: int,
        L: int,
        C: int,
    ) -> None:
        with self._lock:
            for K, ((content, tool_calls, message), results) in enumerate(
                zip(response_list, results_list), 1
            ):
                coordinate = SearchCoordinate(R=R, C=C, L=L, K=K)
                node = SearchNode(
                    run_id=self.run_id,
                    node_id=self.node_id(R=R, C=C, L=L, K=K),
                    node_label=self.node_label(R=R, C=C, L=L, K=K),
                    coordinate=coordinate,
                    parents=parents,
                    created_at=self.now(),
                    core=self._build_core(content, tool_calls, results, message),
                    detail={
                        "prompt": json_value(prompt),
                        "message": json_value(message),
                        "content": content,
                        "tool_calls": json_value(tool_calls),
                        "tool_results": json_value(results),
                        "usage": json_value(usage),
                    },
                )
                self.nodes[node.node_id] = node
                if self._nodes_path is not None:
                    with self._nodes_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(node.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
                        )

    def push_candidate(self, candidate: CandidateRecord) -> bool:
        priority = self._candidate_priority(candidate)
        if priority is None:
            return False
        with self._lock:
            self._candidates.append(candidate)
        return True

    def update_diagnostics(self, formula: str, diagnostics: dict[str, Any]) -> None:
        with self._lock:
            for candidate in self._candidates:
                if candidate.formula == formula:
                    candidate.details["eic_diagnostics"] = diagnostics

    def ranked_candidates(self) -> list[CandidateRecord]:
        with self._lock:
            return sorted(self._candidates, key=self._candidate_priority)

    def pareto_indices(self, candidates: list[CandidateRecord] | None = None) -> list[int]:
        candidates = candidates if candidates is not None else self.ranked_candidates()
        indices = []
        current_complexity = float("inf")
        for index, candidate in enumerate(candidates):
            try:
                complexity = float(candidate.complexity)
            except (TypeError, ValueError):
                complexity = float("inf")
            if complexity < current_complexity:
                indices.append(index)
                current_complexity = complexity
        return indices

    def result(self, status: str, progress: str) -> SearchResult:
        candidates = self.ranked_candidates()
        result = SearchResult(
            status=status,
            progress=progress,
            candidates=candidates,
            pareto_front=self.pareto_indices(candidates),
            best_candidate=0 if candidates else None,
        )
        if self._result_path is not None:
            self._write_json(self._result_path, result.to_dict())
        return result

    def records(self, *, include_detail: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            nodes = list(self.nodes.values())
        return [node.to_dict(include_detail=include_detail) for node in nodes]

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self.nodes)

    @property
    def latest_coordinate(self) -> SearchCoordinate | None:
        """Return the coordinate of the most recently recorded search node."""
        with self._lock:
            if not self.nodes:
                return None
            node_id = next(reversed(self.nodes))
            return self.nodes[node_id].coordinate

    def node_record(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            node = self.nodes.get(node_id)
        return node.to_dict() if node is not None else None

    def _candidate_priority(self, candidate: CandidateRecord) -> tuple[float, float] | None:
        validation = candidate.metric(self.ranking_metric, "validation")
        metric = validation if validation is not None else candidate.metric(
            self.ranking_metric, "train"
        )
        try:
            metric = float(metric)
            complexity = float(candidate.complexity)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(metric) or not math.isfinite(complexity):
            return None
        return (-metric if self.larger_is_better else metric, complexity)

    @staticmethod
    def _build_core(content, tool_calls, results, message) -> dict[str, Any]:
        best_record = None
        is_candidate = False
        for tool_call, call_result in zip(tool_calls, results):
            if not call_result.ok:
                continue
            split_results = call_result.result.get("data_split_results", {})
            split_name = "validation" if "validation" in split_results else "train"
            metrics = split_results.get(split_name, {}).get("metrics")
            if metrics is None:
                continue
            record = {
                "formula": call_result.result.get("formula"),
                "split": split_name,
                "rmse": metrics.get("rmse"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
            }
            if (
                best_record is None
                or SearchRunState._score_key(record)
                < SearchRunState._score_key(best_record)
            ):
                best_record = record
            is_candidate = is_candidate or bool(call_result.result.get("is_candidate"))
        return {
            "formula": best_record.pop("formula") if best_record else None,
            "score": best_record,
            "is_candidate": is_candidate,
            "tool_names": [tool_call.name for tool_call in tool_calls],
            "tool_count": len(tool_calls),
            "content_preview": SearchRunState._build_preview(content, tool_calls, message),
        }

    @staticmethod
    def _score_key(record: dict[str, Any]) -> float:
        try:
            score = float(record.get("mse"))
        except (TypeError, ValueError):
            return float("inf")
        return score if math.isfinite(score) else float("inf")

    @staticmethod
    def _build_preview(content, tool_calls, message, max_param_length: int = 240) -> str:
        parts = []
        if not isinstance(message, dict):
            reason = ""
        elif message.get("reasoning"):
            reason = str(message["reasoning"])
        elif isinstance(details := message.get("reasoning_details"), list):
            reason = "\n".join(
                str(item.get("text"))
                for item in details
                if isinstance(item, dict) and item.get("text")
            )
        else:
            reason = ""
        if reason:
            parts.append(f"Reason:\n{reason.strip()}")
        if content:
            parts.append(f"Content:\n{content.strip()}")
        if tool_calls:
            lines = []
            for index, tool_call in enumerate(tool_calls, 1):
                params = SearchRunState._truncate(tool_call.params or {}, max_param_length)
                lines.append(
                    f"{index:02d}. {tool_call.name}({json.dumps(params, ensure_ascii=False)})"
                )
            parts.append("Tool Calls:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    @staticmethod
    def _truncate(value: Any, max_length: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_length else value[:max_length] + "...<truncated>"
        if isinstance(value, list):
            return [SearchRunState._truncate(item, max_length) for item in value]
        if isinstance(value, dict):
            return {
                key: SearchRunState._truncate(item, max_length)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
