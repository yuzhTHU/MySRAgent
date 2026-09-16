#!/usr/bin/env python3
"""Run the preserved LSR-Synth status summarizer for the current paired experiment."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE = REPO_ROOT / "scripts/summarize_lsr_synth_run_core.pyc"
CURRENT_MANIFEST = (
    REPO_ROOT / "logs/bench_sr_agent/"
    "lsr_synth_r2c1l10k1_seed26091320_20260915_commit0c98f4d_oodsplit/"
    "controller/manifest.json"
)


def main() -> int:
    args = list(sys.argv[1:])
    if "--manifest" not in args:
        args[0:0] = ["--manifest", str(CURRENT_MANIFEST)]
    return subprocess.call([sys.executable, str(CORE), *args], cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
