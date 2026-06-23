from __future__ import annotations

import json

import numpy as np
import pytest

from sr_agent.cli.tool import build_argparser, load_context, load_params, main


def run_cli(argv: list[str]) -> None:
    main(build_argparser().parse_args(argv))


def test_load_context_from_explicit_data_field(tmp_path):
    path = tmp_path / "context.npz"
    data = {
        "x1": np.array([1.0, 2.0, 3.0]),
        "y": np.array([2.0, 4.0, 6.0]),
    }
    np.savez(path, data=data, target="y")

    context = load_context(path)

    assert context["target"] == "y"
    assert set(context["data"]) == {"x1", "y"}
    np.testing.assert_allclose(context["data"]["x1"], [1.0, 2.0, 3.0])


def test_load_context_from_variable_fields(tmp_path):
    path = tmp_path / "context.npz"
    np.savez(
        path,
        x1=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 4.0, 6.0]),
        target="y",
    )

    context = load_context(path)

    assert context["target"] == "y"
    assert set(context["data"]) == {"x1", "y"}


def test_load_params_merges_file_then_inline_json(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"f": "x1", "fit": False}), encoding="utf-8")

    assert load_params('{"fit": true}', str(path)) == {"f": "x1", "fit": True}


def test_call_outputs_formatted_result_only(tmp_path, capsys):
    context_path = tmp_path / "context.npz"
    np.savez(
        context_path,
        x1=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 4.0, 6.0]),
        target="y",
    )

    run_cli(
        [
            "call",
            "statistics_analysis",
            "--context",
            str(context_path),
            "--params",
            '{"variables": ["x1"]}',
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("Variable 'x1':")
    assert "ToolCallResult" not in captured.out
    assert captured.err == ""


def test_call_returns_nonzero_on_tool_error(tmp_path, capsys):
    context_path = tmp_path / "context.npz"
    np.savez(
        context_path,
        x1=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0]),
        target="y",
    )

    with pytest.raises(SystemExit) as exc:
        run_cli(
            [
                "call",
                "evaluate_formula",
                "--context",
                str(context_path),
                "--params",
                '{"f": "invalid_syntax!!"}',
            ]
        )

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "SyntaxError" in captured.out
