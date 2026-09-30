# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Command-line gateway for sr-agent tools."""
from __future__ import annotations

import json
import argparse
import numpy as np
import sr_agent.tools
from pathlib import Path
from typing import Any
from sr_agent.tools import BaseTool


def load_json_text(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Tool params must be a JSON object.")
    return value


def load_params(params: str | None = None, params_file: str | None = None) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if params_file:
        loaded |= load_json_text(Path(params_file).read_text(encoding="utf-8"))
    if params:
        loaded |= load_json_text(params)
    return loaded


def decode_npz_value(value: np.ndarray) -> Any:
    if value.shape == ():
        return value.item()
    return value


def load_context(path: str | Path, target: str | None = None) -> dict[str, Any]:
    """Load a BaseTool context from context.npz."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    with np.load(path, allow_pickle=True) as archive:
        context = {key: decode_npz_value(archive[key]) for key in archive.files}

    if target is not None:
        context["target"] = target

    if "data" in context:
        data = context["data"]
        if not isinstance(data, dict):
            raise ValueError('context.npz field "data" must contain a dict[str, np.ndarray].')
        if "target" not in context:
            raise ValueError('context.npz with a "data" field must also contain a "target" field.')
        context["target"] = str(context["target"])
        return context

    if "target" not in context:
        raise ValueError(
            'context.npz must either contain fields "data" and "target", '
            'or contain a scalar "target" naming the target variable.'
        )

    target_name = str(context["target"])
    data = {
        key: value
        for key, value in context.items()
        if key != "target" and isinstance(value, np.ndarray)
    }
    if target_name not in data:
        raise ValueError(
            f'Target variable "{target_name}" is not present as an array field in context.npz.'
        )
    return {"data": data, "target": target_name}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sr-agent tools from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered tool names.")
    list_parser.add_argument("--json", action="store_true", help="Output tool names as JSON.")

    schema_parser = subparsers.add_parser("schema", help="Print a tool schema.")
    schema_parser.add_argument("tool", nargs="?", help="Tool name. Omit to print all schemas.")

    call_parser = subparsers.add_parser("call", help="Call one registered tool.")
    call_parser.add_argument("tool", help="Tool name.")
    call_parser.add_argument("--context", default="context.npz", help="Path to context.npz containing BaseTool context fields.")
    call_parser.add_argument("--target", default=None, help="Override context target name.")
    call_parser.add_argument("--params", default=None, help='Tool parameters as a JSON object, e.g. \'{"f": "sin(x1)"}\'.')
    call_parser.add_argument("--params-file", default=None, help="JSON file with tool parameters.")
    return parser


def tool_class(name: str) -> type[BaseTool]:
    return BaseTool.create(name, create_instance=False)


def main(argv: list[str] | argparse.Namespace | None = None) -> None:
    parser = build_argparser()
    args = argv if isinstance(argv, argparse.Namespace) else parser.parse_args(argv)

    if args.command == "list":
        if args.json:
            print(json.dumps(list(BaseTool.REGISTRY_DICT), indent=2, ensure_ascii=False))
        else:
            for idx, (name, tool_cls) in enumerate(BaseTool.REGISTRY_DICT.items()):
                description = (tool_cls.metadata.description or "").strip()
                print(f"[{idx:02d}] {name}: {description}")
    elif args.command == "schema":
        schema = tool_class(args.tool).to_dict() if args.tool else BaseTool.to_tool_list()
        print(json.dumps(schema, indent=2, ensure_ascii=False))
    elif args.command == "call":
        tool_cls = tool_class(args.tool)
        context = load_context(args.context, target=args.target)
        params = load_params(args.params, args.params_file)
        result = tool_cls(**context)(**params)
        print(result.result_str)
        if not result.ok:
            raise SystemExit(1)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
