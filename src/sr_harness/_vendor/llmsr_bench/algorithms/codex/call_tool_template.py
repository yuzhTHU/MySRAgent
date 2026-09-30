# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Problem-local tool wrapper for this Codex symbolic-regression run."""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from sr_agent.tools import BaseTool
from sr_agent.cli.tool import load_context, load_params, build_argparser
from sr_agent._vendor.llmsr_bench.algorithms.codex.utils import record_tool_call


WORK_DIR = Path("<WORK_DIR>")
CONTEXT_PATH = Path("<CONTEXT_PATH>")
TOOL_CALL_LOG_PATH = Path("<TOOL_CALL_LOG_PATH>")
ENABLED_TOOLS = "<ENABLED_TOOLS>".split(',')

os.chdir(WORK_DIR)
os.environ.setdefault("HF_HOME", "/tmp/sr_agent_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/sr_agent_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sr_agent_mplconfig")
for cache_dir in ("HF_HOME", "HF_DATASETS_CACHE", "MPLCONFIGDIR"):
    Path(os.environ[cache_dir]).mkdir(parents=True, exist_ok=True)


def main(args) -> None:
    if args.command == "list":
        for idx, metadata in enumerate(BaseTool.load_tool_list(ENABLED_TOOLS)):
            name = metadata['name']
            description = '\n'.join('  ' + line for line in (metadata['description'] or "").splitlines())
            print(f"{name}\n{description}\n")
    elif args.command == "schema":
        if not args.tool:
            schema = BaseTool.to_tool_list(ENABLED_TOOLS)
            print(json.dumps(schema, indent=2, ensure_ascii=False))
        elif args.tool in ENABLED_TOOLS:
            tool_cls = BaseTool.create(args.tool, create_instance=False)
            schema = tool_cls.to_dict()
            print(json.dumps(schema, indent=2, ensure_ascii=False))
        else:
            print(f"Tool {args.tool!r} is not in the enabled tools list: {ENABLED_TOOLS}", file=sys.stderr)
    elif args.command == "call":
        if args.tool not in ENABLED_TOOLS:
            print(f"Tool {args.tool!r} is not in the enabled tools list: {ENABLED_TOOLS}", file=sys.stderr)
            raise SystemExit(1)
        else:
            tool_cls = BaseTool.create(args.tool, create_instance=False)
            context = load_context(args.context, target=args.target)
            params = load_params(args.params, args.params_file)
            result = tool_cls(**context)(**params)
            record_tool_call(TOOL_CALL_LOG_PATH, args, params, result)
            print(result.result_str)
        if not result.ok:
            raise SystemExit(1)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.context = CONTEXT_PATH
    main(args)
