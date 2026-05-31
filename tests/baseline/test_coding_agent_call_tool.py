from __future__ import annotations

import json

import numpy as np
import pytest

from baseline.coding_agent.call_tool import log_tool_call
from baseline.coding_agent.call_tool import main as baseline_main
from sr_agent.cli.tool import main


def test_baseline_list_prints_tool_names_and_descriptions(capsys):
    with pytest.raises(SystemExit) as exc:
        baseline_main(["list"])

    output = capsys.readouterr().out
    assert exc.value.code == 0
    assert "evaluate_formula:" in output
    assert "Evaluate formula fit quality to data." in output


def test_baseline_call_tool_logs_result_next_to_manifest_result(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context_path = run_dir / "context.npz"
    result_path = tmp_path / "logs" / "demo_result.json"
    log_path = tmp_path / "logs" / "demo_result.tool_calls.jsonl"
    manifest_path = run_dir / "manifest.json"
    np.savez(
        context_path,
        x1=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 4.0, 6.0]),
        target="y",
    )
    manifest_path.write_text(
        json.dumps({"result_path": str(result_path), "tool_call_log_path": str(log_path)}),
        encoding="utf-8",
    )

    main(
        [
            "call",
            "statistics_analysis",
            "--context",
            str(context_path),
            "--params",
            '{"variables": ["x1"]}',
        ],
        on_tool_result=log_tool_call,
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("Variable 'x1':")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["tool"] == "statistics_analysis"
    assert records[0]["params"] == {"variables": ["x1"]}
    assert records[0]["tool_call_result"]["ok"] is True
    assert "Variable 'x1'" in records[0]["tool_call_result"]["result_str"]
