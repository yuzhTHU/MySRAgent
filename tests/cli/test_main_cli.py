from __future__ import annotations

import pytest

from sr_harness.cli import entrypoint, main, setup_parser


def test_main_help_lists_subcommands(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "SRHarness command-line interface" in output
    assert "run" in output
    assert "bench" in output


def test_bare_command_prints_help(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness"])
    assert entrypoint() == 0
    output = capsys.readouterr().out
    assert "SRHarness command-line interface" in output
    assert "run              Run SRAgent" in output
    assert "bench            Evaluate an algorithm" in output


def test_run_help_is_delegated(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "run", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: sr-harness run" in output
    assert "--web" not in output
    assert "--anonymize" not in output


def test_main_dispatches_parsed_namespace():
    parser = setup_parser()
    args = parser.parse_args(["tool", "list", "--json"])
    assert main(args) == 0
