import argparse
import json

import numpy as np

from sr_agent._vendor.llmsr_bench.algorithms import get_algorithm
from sr_agent._vendor.llmsr_bench.algorithms import my_igsr


def test_dispatch_and_default_model():
    args = my_igsr.update_parser(argparse.ArgumentParser()).parse_args([])
    assert get_algorithm("my_igsr") is my_igsr.run
    assert args.llm_model == "deepseek/deepseek-v4-flash-0731"
    assert args.igsr_total_budget == 5
    assert args.igsr_tree is True


def test_relative_worker_python_is_accepted_by_parser():
    args = my_igsr.update_parser(argparse.ArgumentParser()).parse_args(["--igsr_python", "venv/bin/python"])
    assert args.igsr_python == "venv/bin/python"


def test_openrouter_request_controls_are_parsed():
    args = my_igsr.update_parser(argparse.ArgumentParser()).parse_args(
        ["--igsr_max_tokens", "8192", "--igsr_reasoning", "low",
         "--no-igsr_error_on_all_weights_zero"])
    assert args.igsr_max_tokens == 8192
    assert args.igsr_reasoning == "low"
    assert args.igsr_error_on_all_weights_zero is False


def test_result_predicts_and_restores_symbols():
    result = my_igsr._build_result(
        {"terms": ["x1", "np.square(x2)"], "coefficients": [2.0, -0.5], "intercept": 3.0},
        ["y", "mass", "time"],
    )
    np.testing.assert_allclose(result.predict(np.array([[2.0, 4.0], [-1.0, 2.0]])), [-1.0, -1.0])
    assert "mass" in result.expression and "time" in result.expression
    assert "np." not in result.expression


def test_result_predicts_bare_abs_term():
    result = my_igsr._build_result(
        {"terms": ["abs(x1)"], "coefficients": [2.0], "intercept": 0.0},
        ["y", "x"],
    )
    np.testing.assert_allclose(result.predict(np.array([[-3.0], [4.0]])), [6.0, 8.0])


def test_usage_summary_survives_error_record(tmp_path):
    path = tmp_path / "usage.jsonl"
    rows = [
        {"status": "ok", "elapsed_seconds": 1.5,
         "usage": {"token": {"prompt": 10, "answer": 3}, "price": {"total": 0.01}}},
        {"status": "error", "elapsed_seconds": 2.0, "usage": {"token": {}, "price": {}}},
        {"status": "ok", "elapsed_seconds": 0.5,
         "usage": {"token": {"prompt": 7, "reason": 2}, "price": {"total": 0.02}}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = my_igsr._summarize_usage(path)
    assert summary == {"calls": 2, "failed_calls": 1, "elapsed_seconds": 4.0,
                       "token": {"prompt": 17, "answer": 3, "reason": 2},
                       "price": {"total": 0.03}}
