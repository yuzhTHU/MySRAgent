from __future__ import annotations

import threading

import numpy as np
import pytest

from sr_harness.runtime import InteractionController
from sr_harness.web.app import create_app
from sr_harness.runtime import ModelRouter
from sr_harness.core import SearchRunState, ToolCall
from sr_harness import SRAgent
from sr_harness.tools.evaluate_formula import EvaluateTool
from sr_harness.utils import ParallelTimer
from sr_harness.skills import SkillManager
from sr_harness.tools.eic import EICTool
from sr_harness.tools.nd2 import ND2Tool
from sr_harness.tools.sr4mdl import SR4MDLTool
from sr_harness.tools.subagent import SubagentTool
from sr_harness.tools.web_research import WebSearchTool


def test_eic_is_reproducible_and_finite():
    tool = EICTool(data={"x": np.linspace(1.0, 2.0, 64)})
    first = tool.execute("x + x**2", repeats=4, random_state=7)
    second = tool.execute("x + x**2", repeats=4, random_state=7)
    assert first["eic"] == second["eic"]
    assert np.isfinite(first["eic"])


def test_eic_reports_unstable_masked_subtree_in_annotated_tree():
    x = np.logspace(-12, -2, 128)
    tool = EICTool(data={"x": x})
    result = tool.execute("0*(sqrt(1+x)-1)+x", repeats=4, random_state=7)
    assert result["eic"] > result["output_eic"] + 3
    assert result["worst_subtree"] != "root"
    assert "likely redundant subtree" in result["tree"]
    assert "← EIC=" in result["tree"]


def test_eic_result_can_reenter_candidate_search_state():
    values = np.linspace(1.0, 2.0, 32)
    result = EICTool(
        data={"x": values, "y": values + values**2},
        target="y",
    ).execute("x + x**2", repeats=2)
    assert result["is_candidate"] is True
    assert result["data_split_results"]["train"]["metrics"]["mse"] == 0.0
    assert result["eic_diagnostics"]["worst_subtree"] == result["worst_subtree"]


def test_new_candidate_is_audited_without_spending_an_llm_round():
    values = np.linspace(1.0, 2.0, 32)
    context = {"data": {"x": values, "y": values}, "target": "y"}
    candidate = EvaluateTool(**context)(f="x", show_diagnostics=False)
    agent = object.__new__(SRAgent)
    agent.ranking_metric = "mse"
    agent.larger_is_better = False
    agent.tools = [EICTool(**context)]
    agent.tools_counter = ParallelTimer(unit="call")
    agent.save_path = None
    agent.run_state = SearchRunState(
        save_path=None,
        ranking_metric="mse",
        larger_is_better=False,
    )
    topk = agent.collect_candidates(
        [("", [ToolCall(name="evaluate_formula", params={})], {})],
        [[candidate]],
        R=1,
        L=1,
        C=1,
    )
    assert topk[0].details["eic_diagnostics"]["worst_subtree"] == "root"
    assert agent.tools_counter.named_count["evaluate_eic"] == 1


def test_subagent_uses_isolated_callback_prompt():
    captured = {}

    def callback(messages):
        captured["messages"] = messages
        return "independent result"

    result = SubagentTool(
        data={"x": np.arange(3.0), "y": np.arange(3.0)},
        target="y",
        subagent_callback=callback,
    ).execute(
        "Critique this hypothesis.",
        mode="candidate_critique",
        candidate_formulas=["x"],
    )
    assert result["content"] == "independent result"
    assert "pearson_to_target=1" in captured["messages"][1]["content"]
    assert "Candidate formulas:\n- x" in captured["messages"][1]["content"]


def test_web_search_callback_is_bounded():
    tool = WebSearchTool(web_search_callback=lambda query, limit: [
        {"title": str(i), "url": f"https://example.com/{i}", "snippet": query}
        for i in range(20)
    ])
    result = tool.execute("symbolic regression", max_results=3)
    assert len(result["results"]) == 3


def test_eic_doc_is_registered_as_runtime_skill(tmp_path):
    manager = SkillManager(
        built_in_directory="src/sr_harness/skills",
        custom_directory=tmp_path / "custom",
    )
    manager.register_tool_docs([EICTool, ND2Tool, SR4MDLTool])
    names = manager.load_skills()
    assert "eic-structural-stability" in names
    assert "nd2-network-dynamics" in names
    assert "sr4mdl-search" in names


def test_sr4mdl_constructs_search_data_like_pysr(tmp_path, monkeypatch):
    root = tmp_path / "SR4MDL"
    root.mkdir()
    (root / "regressor.py").touch()
    checkpoint = root / "checkpoint.pth"
    checkpoint.touch()
    monkeypatch.setenv("SR4MDL_HOME", str(root))
    monkeypatch.setenv("SR4MDL_CHECKPOINT", str(checkpoint))
    captured = {}

    def fake_search(self, **kwargs):
        captured.update(kwargs)
        return "x1 + x2"

    monkeypatch.setattr(SR4MDLTool, "_run_sr4mdl", fake_search)
    values = np.linspace(0.0, 1.0, 32)
    result = SR4MDLTool(
        data={"a": values, "b": values**2, "y": values + values**2},
        target="y",
        train_indices=np.arange(24),
        validation_indices=np.arange(24, 32),
    ).execute(["+", "*"], [], x=["a", "b"], y="y", n_iter=7)

    assert set(captured["X"]) == {"x1", "x2"}
    assert captured["n_iter"] == 7
    assert result["formula"] == "a + b"
    assert result["method"] == "SR4MDL-MCTS"


def test_nd2_returns_search_candidate_instead_of_export(tmp_path, monkeypatch):
    root = tmp_path / "ND2"
    root.mkdir()
    (root / "search.py").touch()
    checkpoint = root / "checkpoint.pth"
    checkpoint.touch()
    monkeypatch.setenv("ND2_HOME", str(root))
    monkeypatch.setenv("ND2_CHECKPOINT", str(checkpoint))

    def fake_search(self, **kwargs):
        return {
            "formula": "omega + aggr(sin(sour(x) - targ(x)))",
            "prefix": ["add", "omega", "aggr", "sin", "sub", "sour", "x", "targ", "x"],
            "train_metrics": {"mse": 0.0, "rmse": 0.0, "r2": 1.0, "complexity": 9.0},
            "validation_metrics": None,
        }

    monkeypatch.setattr(ND2Tool, "_run_nd2", fake_search)
    adjacency = np.array([[0, 1], [1, 0]])
    x = np.arange(12.0).reshape(6, 2)
    result = ND2Tool(
        data={"A": adjacency, "x": x, "omega": np.ones_like(x), "dx": x},
        target="dx",
    ).execute(vars_node=["x", "omega"], y="dx")
    assert result["is_candidate"] is True
    assert result["method"] == "ND2-NDformer-MCTS"
    assert result["data_split_results"]["train"]["metrics"]["mse"] == 0.0


def test_network_split_preserves_graph_axes():
    agent = object.__new__(SRAgent)
    agent.validation_fraction = 0.25
    agent.split_by = "random"
    agent.split_random_state = 7
    adjacency = np.array([[0, 1], [1, 0]])
    edges = np.array([[0, 1], [1, 0]])
    x = np.arange(16.0).reshape(8, 2)
    train, validation = agent._split_data(
        {"A": adjacency, "G": edges, "x": x},
        {"dx": x + 1},
    )
    assert train["x"].shape == (6, 2)
    assert validation["x"].shape == (2, 2)
    assert np.array_equal(train["A"], adjacency)
    assert np.array_equal(validation["G"], edges)


def test_model_router_uses_base_for_simple_task_and_strong_for_complex_task():
    router = ModelRouter(
        enabled=True,
        base_provider="openrouter",
        base_model="cheap",
        strong_model="strong",
    )
    score, reasons = router.assess("Find y = f(x).", feature_count=1)
    assert router.route(task_score=score, task_reasons=reasons, refinement_step=1).tier == "base"
    score, reasons = router.assess("Discover a noisy network dynamics ODE.", feature_count=5)
    assert router.route(task_score=score, task_reasons=reasons, refinement_step=1).tier == "strong"


def test_model_router_escalates_stagnated_search_and_respects_disable():
    router = ModelRouter(
        enabled=True,
        base_provider="base-provider",
        base_model="base-model",
        strong_provider="strong-provider",
        strong_model="strong-model",
    )
    route = router.route(task_score=0, task_reasons=[], refinement_step=3)
    assert (route.tier, route.provider, route.model) == (
        "strong", "strong-provider", "strong-model"
    )
    router.enabled = False
    assert router.route(task_score=9, task_reasons=[], refinement_step=4).tier == "base"


def test_subagent_prompt_contains_grounded_sr_mandate():
    captured = {}

    def callback(messages):
        captured["messages"] = messages
        return "review"

    SubagentTool(subagent_callback=callback).execute(
        "Check floating-point cancellation and recommend a discriminating test.",
        mode="candidate_critique",
        candidate_formulas=["exp(x)-1", "x"],
        evidence="Both formulas fit the observed interval.",
    )
    prompt = captured["messages"][1]["content"]
    assert "Available main-agent tools" in prompt
    assert "candidate_critique" in prompt
    assert "evaluate_eic" in prompt


def test_interaction_controller_round_trip_and_commands():
    controller = InteractionController()
    received = {}

    thread = threading.Thread(
        target=lambda: received.setdefault("answer", controller.ask("Choose?", timeout=2))
    )
    thread.start()
    while not controller.events():
        thread.join(0.01)
    question = controller.events()[0]
    controller.reply(question["id"], "continue")
    thread.join(2)
    assert received["answer"] == "continue"

    controller.command("message", "try a power law")
    assert controller.checkpoint() == ["try a power law"]
    controller.command("pause")
    assert controller.status()["paused"] is True
    controller.command("resume")
    assert controller.status()["paused"] is False


def test_create_app_requires_explicit_controller(tmp_path):
    with pytest.raises(TypeError):
        create_app(tmp_path)
    app = create_app(tmp_path, controller=InteractionController())
    assert isinstance(app.state.controller, InteractionController)
