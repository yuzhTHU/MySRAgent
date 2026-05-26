# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import uuid
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from ..api.core import ToolCall
from ..tools import ToolCallResult


class SearchRecordWriter:
    """Append-only visualization records for the search tree."""
    def __init__(self, save_path: str | None, agent: "SRAgent"):
        self.enabled = save_path is not None
        if not self.enabled:
            return
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.save_path / "records.jsonl"
        self.manifest_path = self.save_path / "manifest.json"
        self.tree_id = uuid.uuid4().hex
        self.seq = 0
        if self.records_path.exists():
            with open(self.records_path, "r", encoding="utf-8") as f:
                self.seq = sum(1 for line in f if line.strip())
        if not self.manifest_path.exists():
            manifest = {
                "schema_version": 1,
                "created_at": self.now(),
                "record_file": "records.jsonl",
                "agent": {
                    "class": agent.__class__.__name__,
                    "llm_provider": agent.llm_provider,
                    "llm_model": agent.llm_model,
                    "tool_parser": str(agent.tool_parser),
                    "local_sample_size": agent.local_sample_size,
                    "max_refinement_depth": agent.max_refinement_depth,
                    "global_width": agent.global_width,
                    "max_restart_loop": agent.max_restart_loop,
                    "restart_top_k": agent.restart_top_k,
                    "max_workers": agent.max_workers,
                },
            }
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.serialization(manifest), ensure_ascii=False, indent=2) + '\n')

    def node_id(self, R: int, C: int, L: int, K: int) -> str:
        return f"{self.tree_id}:{self.node_label(R=R, C=C, L=L, K=K)}"

    def node_label(self, R: int, C: int, L: int, K: int) -> str:
        return f"R{R}-C{C}-L{L}-K{K}"

    def node_parent(self, node_id: str, kind: str, strength: str) -> dict[str, str]:
        return {"node_id": node_id, "kind": kind, "strength": strength}

    def record_iteration(
        self,
        response_list: list,
        results_list: list,
        parents: List[Dict[str, Any]],
        prompt: List[Dict[str, Any]],
        usage: Dict[str, Any],
        R: int,
        L: int,
        C: int,
    ) -> None:
        if not self.enabled:
            return
        with open(self.records_path, "a", encoding="utf-8") as f:
            for K, ((content, tool_calls, message), results) in enumerate(zip(response_list, results_list), 1):
                record = {
                    "schema_version": 1,
                    "seq": self.seq + 1,
                    "run_id": self.save_path.name,
                    "tree_id": self.tree_id,
                    "node_id": self.node_id(R=R, C=C, L=L, K=K),
                    "node_label": self.node_label(R=R, C=C, L=L, K=K),
                    "created_at": self.now(),
                    "coord": {"R": R, "C": C, "L": L, "K": K},
                    "progress": f"(R={R}) x (C={C}) x (L={L}) x (K={K})",
                    "parents": parents,
                    "core": self.build_core(content, tool_calls, results, message),
                    "detail": {
                        "prompt": self.serialization(prompt),
                        "message": self.serialization(message),
                        "content": content,
                        "tool_calls": self.serialization(tool_calls),
                        "tool_results": self.serialization(results),
                        "usage": self.serialization(usage),
                    },
                }
                f.write(json.dumps(self.serialization(record), ensure_ascii=False) + '\n')
                self.seq += 1

    def now(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    def build_core(
        self,
        content: str,
        tool_calls: list[ToolCall],
        results: list[ToolCallResult],
        message: dict[str, Any],
    ) -> dict[str, Any]:
        best_record = None
        is_candidate = False
        for tool_call, call_result in zip(tool_calls, results):
            if not call_result.ok:
                continue
            if (metrics := call_result.result.get("metrics")) is None:
                continue
            record = {
                "formula": call_result.result.get("formula"),
                "rmse": metrics.get("rmse"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
            }
            if best_record is None or record["mse"] < best_record["mse"]:
                best_record = record
            is_candidate = is_candidate or bool(call_result.result.get("is_candidate"))
        return {
            "formula": best_record.pop('formula') if best_record else None,
            "score": best_record if best_record else None,
            "is_candidate": is_candidate,
            "tool_names": [tool_call.name for tool_call in tool_calls],
            "tool_count": len(tool_calls),
            "content_preview": self.build_preview(content, tool_calls, message),
        }

    def build_preview(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
        message: dict[str, Any] | None,
        max_param_length: int = 240,
    ) -> str:
        parts = []

        # Reason Part
        if not isinstance(message, dict):
            reason = ""
        elif message.get("reasoning"):
            reason = str(message["reasoning"])
        elif isinstance(details := message.get("reasoning_details"), list):
            reason = "\n".join([
                str(item.get("text")) for item in details 
                if isinstance(item, dict) and item.get("text")
            ])
        else:
            reason = ""
        if reason:
            parts.append(f"Reason:\n{reason.strip()}")

        # Content Part
        if content:
            parts.append(f"Content:\n{content.strip()}")

        # Tool Calls Part
        if tool_calls:
            tool_lines = []
            for idx, tool_call in enumerate(tool_calls, 1):
                params = self.truncate(tool_call.params or {}, max_param_length)
                params_str = json.dumps(params, ensure_ascii=False)
                tool_lines.append(f"{idx:02d}. {tool_call.name}({params_str})")
            parts.append("Tool Calls:\n" + "\n".join(tool_lines))
        return "\n\n".join(parts)

    def truncate(self, value: Any, max_length: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_length else value[:max_length] + "...<truncated>"
        elif isinstance(value, list):
            return [self.truncate(item, max_length) for item in value]
        elif isinstance(value, dict):
            return {key: self.truncate(item, max_length) for key, item in value.items()}
        else:
            return value

    def serialization(self, value: Any) -> Any:
        """Best-effort conversion for JSONL records."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, ToolCall):
            return {
                "name": value.name,
                "params": self.serialization(value.params),
                "id": value.id,
                "raw": self.serialization(value.raw),
                "raw_str": value.raw_str,
            }
        if isinstance(value, ToolCallResult):
            return {
                "ok": value.ok,
                "result": self.serialization(value.result),
                "result_str": value.result_str,
                "meta_data": self.serialization(value.meta_data),
            }
        if isinstance(value, dict):
            return {str(k): self.serialization(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.serialization(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
