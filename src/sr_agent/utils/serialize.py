# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Any
from pathlib import Path
from datetime import datetime
from collections.abc import Mapping


def serialize(args: Any):
    if args is None or isinstance(args, (str, int, float, bool)):
        return args
    if isinstance(args, Path):
        return str(args)
    if isinstance(args, datetime):
        return args.isoformat(timespec="seconds")
    if isinstance(args, Mapping):
        return {str(key): serialize(value) for key, value in args.items()}
    if isinstance(args, (list, tuple)):
        return [serialize(value) for value in args]
    if isinstance(args, set):
        return [serialize(value) for value in sorted(args, key=repr)]

    try:
        import numpy as np

        if isinstance(args, np.ndarray):
            return serialize(args.tolist())
        if isinstance(args, np.generic):
            return args.item()
    except ImportError:
        pass

    if hasattr(args, "__dict__"):
        return serialize(vars(args))
    return repr(args)
