import sys
import json
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from sr_agent.tools import ToolCallResult


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def record_tool_call(save_path, args, params: dict[str, Any], result: ToolCallResult) -> None:
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": args.tool,
            "context_path": str(args.context),
            "params": params,
            "tool_call_result": {
                "ok": result.ok,
                "result": result.result,
                "result_str": result.result_str,
                "meta_data": result.meta_data,
            },
        }
        record_str = json.dumps(record, ensure_ascii=False, allow_nan=True, default=json_default)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("a", encoding="utf-8") as f:
            f.write(record_str + "\n")
    except Exception as e:
        print(f"Failed to record tool call: [{type(e).__name__}] {e}", file=sys.stderr)
