import json

import pytest

from sr_agent._vendor.llmsr_bench.algorithms.codex import (
    estimate_credit_usage,
    latest_usage_from_codex_events,
)


def test_estimate_gpt55_credit_usage():
    usage = {
        "input_tokens": 402_661,
        "cached_input_tokens": 352_768,
        "output_tokens": 6_045,
        "total_tokens": 408_706,
    }
    result = estimate_credit_usage("gpt-5.5", usage)
    assert result["billable_tokens"]["uncached_input_tokens"] == 49_893
    assert result["estimated_credits"] == pytest.approx(15.179975)


def test_latest_usage_falls_back_to_rollout(tmp_path, monkeypatch):
    thread_id = "thread-test"
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n")
    rollout_dir = tmp_path / "sessions" / "2026" / "09" / "22"
    rollout_dir.mkdir(parents=True)
    usage = {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2, "total_tokens": 12}
    rollout = rollout_dir / f"rollout-test-{thread_id}.jsonl"
    rollout.write_text(json.dumps({
        "type": "event_msg",
        "payload": {"type": "token_count", "info": {"total_token_usage": usage}},
    }) + "\n")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert latest_usage_from_codex_events(event_path) == usage
