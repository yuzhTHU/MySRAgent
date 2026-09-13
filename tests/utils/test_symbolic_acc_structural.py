from __future__ import annotations

import nd2py as nd
import numpy as np

from sr_agent.utils.symbolic_acc import get_symbolic_acc, llm_judge_equivalence


def test_structural_judge_prompt_rejects_near_fit(monkeypatch):
    seen = {}

    class FakeAPI:
        def __call__(self, messages, **kwargs):
            seen["messages"] = messages
            yield '{"reason": "Taylor approximation", "equivalent": false}', [], {}

    monkeypatch.setattr("sr_agent.api.llm_api.LLMAPI.create", lambda *args, **kwargs: FakeAPI())
    result = llm_judge_equivalence(
        nd.parse("exp(x)"), nd.parse("1+x+x**2/2"),
        {"x": (0.0, 0.01)}, "openrouter", "test-model",
    )

    assert result["equivalent"] is False
    assert "NOT equivalent" in seen["messages"][0]["content"]
    assert "Ground truth: exp(x)\nPredicted:" in seen["messages"][1]["content"]


def test_structural_judge_overrides_numerically_close_approximation(monkeypatch):
    monkeypatch.setattr(
        "sr_agent.utils.symbolic_acc.llm_judge_equivalence",
        lambda **kwargs: {"reason": "different function family", "equivalent": False},
    )
    x = np.linspace(0, 0.01, 100)
    result = get_symbolic_acc(
        nd.parse("exp(x)"), nd.parse("1+x+x**2/2"),
        {"x": x}, llm_judge=True, return_details=True,
    )

    assert result["equivalent"] is False


def test_disagreement_sets_reason_without_human_review(monkeypatch):
    monkeypatch.setattr(
        "sr_agent.utils.symbolic_acc.llm_judge_equivalence",
        lambda **kwargs: {"reason": "different function family", "equivalent": False},
    )
    x = np.linspace(0, 0.01, 100)
    result = get_symbolic_acc(
        nd.parse("exp(x)"), nd.parse("1+x+x**2/2"),
        {"x": x}, llm_judge=True, return_details=True,
    )

    assert result["equivalent"] is False
    assert "automatically accepted LLM judgement" in result["reason"]


def test_unavailable_judge_follows_existing_numeric_fallback(monkeypatch):
    monkeypatch.setattr(
        "sr_agent.utils.symbolic_acc.llm_judge_equivalence",
        lambda **kwargs: {"reason": "unavailable", "equivalent": None},
    )
    x = np.linspace(0, 1, 100)
    result = get_symbolic_acc(
        nd.parse("x"), nd.parse("x"),
        {"x": x}, llm_judge=True, return_details=True,
    )

    assert result["equivalent"] is True
    assert "automatically rejected LLM judgement" in result["reason"]
