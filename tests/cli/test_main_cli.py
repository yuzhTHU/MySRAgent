from __future__ import annotations

import pytest

from sr_harness.cli import entrypoint, main, setup_parser
from sr_harness.cli.run import build_agent_options, run_experiment
from sr_harness.cli.run import setup_parser as setup_run_parser


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
    assert "usage: sr-harness run" in capsys.readouterr().out


def test_main_dispatches_parsed_namespace():
    parser = setup_parser()
    args = parser.parse_args(["tool", "list", "--json"])
    assert main(args) == 0


def test_web_run_receives_the_shared_agent_configuration(monkeypatch, tmp_path):
    args = setup_run_parser().parse_args([])
    args.web = True
    args.save_path = str(tmp_path)
    args.seed = 0
    args.validation_fraction = 0.35
    args.split_by = "random"
    args.split_random_state = 19
    args.force_initial_diagnostics = True
    args.auto_routing = True
    args.strong_llm_provider = "openai"
    args.strong_llm_model = "gpt-5-mini"
    captured = {}

    def fake_serve(log_dir, **kwargs):
        captured.update(log_dir=log_dir, **kwargs)

    monkeypatch.setattr("sr_harness.cli.web.serve", fake_serve)
    expected = build_agent_options(args)

    assert run_experiment(args) is None
    assert {
        name: captured["agent_options"][name]
        for name in expected
    } == expected
    assert captured["agent_options"]["workspace_files"] == args.workspace_files


def test_web_run_rejects_parallel_interactive_tools(tmp_path):
    args = setup_run_parser().parse_args([])
    args.web = True
    args.save_path = str(tmp_path)
    args.seed = 0
    args.max_workers = 2

    with pytest.raises(ValueError, match="require max_workers=0"):
        run_experiment(args)
