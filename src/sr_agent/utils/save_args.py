# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import json
from typing import Any
from pathlib import Path
from datetime import datetime
from collections.abc import MutableMapping
from .serialize import serialize


def save_args(args: Any, args_path: Path, excluded_keys=("saved_time", "command")) -> None:
    """Save args to JSON, backing up an existing file and printing meaningful diffs."""
    args_path = Path(args_path)

    # Serialize args to a dictionary
    if isinstance(args, MutableMapping):
        args["saved_time"] = datetime.now().isoformat(timespec="seconds")
        args_dict = serialize(args)
    else:
        args.saved_time = datetime.now().isoformat(timespec="seconds")
        args_dict = serialize(vars(args))

    if args_path.exists():
        # Backup existing file
        old_args = json.loads(args_path.read_text(encoding="utf-8"))
        if not isinstance(old_args, dict):
            raise ValueError(f"Expected {args_path} to contain a JSON object, got {type(old_args).__name__}")

        try:
            saved_time = old_args.get("saved_time")
            timestamp = datetime.fromisoformat(saved_time.strip())
        except (ValueError, AttributeError):
            timestamp = datetime.fromtimestamp(args_path.stat().st_mtime)
        timestamp = timestamp.strftime('%Y%m%d_%H%M%S')

        backup_path = args_path.with_name(f"{args_path.stem}.{timestamp}{args_path.suffix}")
        i = 1
        while backup_path.exists():
            backup_path = args_path.with_name(f"{args_path.stem}.{timestamp}.{i}{args_path.suffix}")
            i += 1
        args_path.rename(backup_path)

        # Compare Difference
        rows = []
        for key in sorted((set(old_args) | set(args_dict)) - set(excluded_keys)):
            old_value = old_args.get(key, "<missing>")
            new_value = args_dict.get(key, "<missing>")
            try:
                old_text = json.dumps(old_value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                old_text = repr(old_value)
            try:
                new_text = json.dumps(new_value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                new_text = repr(new_value)
            if old_text != new_text:
                rows.append((key, old_text, new_text))

        if rows:
            print(f"Argument differences detected for {args_path}:")
            widths = [
                max(len("key"), *(len(key) for key, old_text, new_text in rows)),
                max(len("old"), *(len(old_text) for key, old_text, new_text in rows)),
                max(len("new"), *(len(new_text) for key, old_text, new_text in rows)),
            ]
            lines = [
                f"{'key':<{widths[0]}}  {'old':<{widths[1]}}  {'new':<{widths[2]}}",
                f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}",
            ] + [
                f"{key:<{widths[0]}}  {old_text:<{widths[1]}}  {new_text:<{widths[2]}}"
                for key, old_text, new_text in rows
            ]
            print("\n".join(lines))

    # Save new args
    args_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
