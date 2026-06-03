# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import sys
import importlib
from pathlib import Path

_ALGO_DIR = Path(__file__).parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))


def list_algorithms() -> list[str]:
    return sorted(
        path.stem for path in Path(__file__).parent.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    )


def get_algorithm(name: str):
    return importlib.import_module(f".{name}", package=__name__)
