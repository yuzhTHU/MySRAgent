from __future__ import annotations

import json
import numpy as np
from types import SimpleNamespace

from sr_agent.api.core import ToolCall
from sr_agent.sr_agent import SRAgent
from sr_agent.tools.base_tool import BaseTool, ToolCallResult, ToolMetadata
from sr_agent.tools.read_skill import ReadSkill


@BaseTool.register("unit_parallel_tool")
class UnitParallelTool(BaseTool):
    metadata = ToolMetadata(name="unit_parallel_tool")

    def execute(self, value: int) -> dict:
        """Return a value with inherited context."""
        return {"value": value, "offset": self.context["offset"]}


@BaseTool.register("unit_messages_tool")
class UnitMessagesTool(BaseTool):
    metadata = ToolMetadata(name="unit_messages_tool")

    def execute(self) -> dict:
        """Return injected messages."""
        return {"messages": self.context["messages"]}


def make_agent(tmp_path):
    agent = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["unit_parallel_tool"],
        save_path=str(tmp_path),
    )
    agent.tools = [UnitParallelTool(offset=10)]
    return agent


def test_agent_exposes_only_skills_from_enabled_tools(tmp_path):
    enabled = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["read_skill", "workspace_shell"],
        save_path=str(tmp_path / "enabled"),
    )
    disabled = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["read_skill"],
        save_path=str(tmp_path / "disabled"),
    )

    enabled_reader = ReadSkill(skill_manager=enabled.skill_manager)
    disabled_reader = ReadSkill(skill_manager=disabled.skill_manager)

    runtime_skill = enabled_reader.skill_manager.load_skills()["workspace-shell"]
    skill_path = runtime_skill.skill_directory / "SKILL.md"
    assert skill_path.name == "SKILL.md"
    assert skill_path.parent.name == "workspace-shell"
    assert skill_path.read_text(encoding="utf-8") == enabled_reader.skill_manager.read_skill("workspace-shell")
    assert "workspace-shell" not in disabled_reader.skill_manager.load_skills()

    assert "<name>workspace-shell</name>" in enabled_reader.metadata.description
    assert "<name>workspace-shell</name>" not in disabled_reader.metadata.description
    result = enabled_reader(name="workspace-shell", file_path="SKILL.md", show_tree=True)
    assert result.ok is True
    assert "Supported commands" in result.result_str
    assert "Skill directory tree:\n- SKILL.md" in result.result_str

    enabled.tools = [enabled_reader]
    parallel_result = enabled.execute_action_parallel(
        [ToolCall(name="read_skill", params={"name": "workspace-shell"})],
        max_workers=2,
    )[0]
    assert parallel_result.ok is True
    assert "# Workspace Shell" in parallel_result.result_str


def test_execute_action_parallel_preserves_order_and_records_usage(tmp_path):
    agent = make_agent(tmp_path)
    actions = [
        ToolCall(name="unit_parallel_tool", params={"value": 1}),
        ToolCall(name="missing_tool", params={}),
        ToolCall(name="unit_parallel_tool", params={"value": 2}),
    ]

    results = agent.execute_action_parallel(actions, max_workers=2)

    assert [result.ok for result in results] == [True, False, True]
    assert results[0].result == {"value": 1, "offset": 10}
    assert results[1].result_str == 'Unknown tool calling for "missing_tool"'
    assert results[2].result == {"value": 2, "offset": 10}
    assert agent.tools_counter.named_count == {"unit_parallel_tool": 2}


def test_get_results_uses_messages_already_injected_into_tool_context(tmp_path):
    agent = make_agent(tmp_path)
    agent.tools = [UnitMessagesTool()]
    agent.max_workers = 2
    prompt = [{"role": "system", "content": "Reusable context"}]
    agent.tools[0].context["messages"] = prompt
    response_list = [
        (
            "",
            [
                ToolCall(name="unit_messages_tool", params={}),
                ToolCall(name="unit_messages_tool", params={}),
            ],
            {},
        )
    ]

    results = agent.get_results(response_list, R=1, L=1, C=1)
    assert results[0][0].result == {"messages": [{"role": "system", "content": "Reusable context"}]}


def test_build_initial_prompt_includes_refinement_budget_rule(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 7

    prompt = agent.build_initial_prompt(
        problem_description="Find y from x.",
        X={"x": [1, 2, 3]},
        y={"y": [2, 4, 6]},
        restart_records=[],
    )

    system_content = prompt[0]["content"]
    assert "at most 7 refinement rounds" in system_content
    assert "At the final refinement round (L=7)" in system_content
    assert "final-answer mechanism" in system_content


def test_split_data_keeps_validation_and_test_out_of_training_context(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2,
        split_random_state=7,
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0)}, {"y": np.arange(10.0) * 2},
    )
    assert len(train["x"]) == 8
    assert len(validation["x"]) == 2
    assert set(train["x"]).isdisjoint(validation["x"])


def test_split_data_ood_uses_high_t_values_for_validation(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="ood",
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0), "t": np.array([8, 1, 5, 0, 9, 3, 7, 2, 6, 4])},
        {"y": np.arange(10.0) * 2},
    )
    assert train["t"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert validation["t"].tolist() == [8, 9]


def test_split_data_ood_falls_back_to_T_and_prefers_t(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.25, split_by="ood",
    )
    _, validation_T = agent._split_data(
        {"x": np.arange(4.0), "T": np.array([300, 500, 200, 400])},
        {"y": np.arange(4.0)},
    )
    _, validation_t = agent._split_data(
        {"t": np.array([4, 1, 3, 2]), "T": np.array([100, 400, 300, 200])},
        {"y": np.arange(4.0)},
    )
    assert validation_T["T"].tolist() == [500]
    assert validation_t["t"].tolist() == [4]


def test_split_data_ood_falls_back_to_first_input_variable(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="ood",
    )
    train, validation = agent._split_data(
        {"x": np.array([8, 1, 5, 0, 9, 3, 7, 2, 6, 4])},
        {"y": np.arange(10.0)},
    )
    assert train["x"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert validation["x"].tolist() == [8, 9]


def test_split_data_random_uses_requested_train_and_validation_sizes(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="random",
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0)}, {"y": np.arange(10.0)},
    )
    assert len(train["x"]) == 8
    assert len(validation["x"]) == 2


def test_split_data_zero_validation_fraction_reuses_training_data(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0, split_by="ood",
    )
    train, validation = agent._split_data(
        {"t": np.arange(10.0)}, {"y": np.arange(10.0)},
    )
    assert len(train["t"]) == 10
    assert len(validation["t"]) == 10
    assert np.array_equal(train["t"], validation["t"])


def test_update_buffer_injects_iteration_status(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_restart_loop = 2
    agent.global_width = 3
    agent.max_refinement_depth = 5
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]
    prompt = list(buffer)
    response_list = [("", [], {"role": "assistant", "content": ""})]
    results_list = [[]]

    updated, _ = agent.update_buffer(
        buffer, response_list, results_list, [], {}, prompt, {}, R=1, L=2, C=1
    )

    assert updated is buffer
    assert len(buffer) == len(prompt) + 1
    assert buffer[0]["content"] == "Base system."
    assert "refinement round L=3/5" in buffer[-1]["content"]
    assert "From now on, 2 refinement round(s) remain" in buffer[-1]["content"]
    assert "No Pareto front yet" in buffer[-1]["content"]
    assert prompt[-1]["content"] == "Solve the task."


def test_update_buffer_injects_current_pareto_front(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 5
    candidate = {
        "formula": "x + y",
        "data_split_results": {
            "train": {"metrics": {"mse": 0.1, "r2": 0.98, "complexity": 3}},
            "validation": {"metrics": {"mse": 0.2, "r2": 0.97, "complexity": 3}},
        },
        "node_id": "R1-C1-L1-K1",
    }
    topk_records = [(*agent.sortby(candidate), 0, candidate)]
    buffer = [{"role": "user", "content": "Find a formula."}]

    agent.update_buffer(
        buffer,
        [("", [], {"role": "assistant", "content": ""})],
        [[]],
        topk_records,
        {},
        list(buffer),
        {},
        R=1,
        L=1,
        C=1,
    )

    status = buffer[-1]["content"]
    assert "No Pareto front yet" not in status
    assert "Validation R²=0.97" in status
    assert "Formula=x + y" in status


def test_update_buffer_final_round_tells_agent_to_submit(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 4
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]
    prompt = list(buffer)
    response_list = [("", [], {"role": "assistant", "content": ""})]
    results_list = [[]]

    agent.update_buffer(buffer, response_list, results_list, [], {}, prompt, {}, R=1, L=4, C=1)
    final_status = buffer[-1]["content"]

    assert "final refinement round" in final_status
    assert "Submit or state your best available target formula now" in final_status
    assert "Do not spend this response on new data exploration" in final_status


def test_record_metric_ignores_non_candidate_tool_results(tmp_path):
    agent = make_agent(tmp_path)
    diagnostic_result = ToolCallResult(
        True,
        {"content": "diagnostic output without formula metrics"},
        "diagnostic output without formula metrics",
        {},
    )

    assert agent.record_metric(diagnostic_result.result) == (None, None)
    assert agent.sortby(diagnostic_result.result) is None


def test_update_buffer_sorts_unwrapped_tool_results(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    monkeypatch.setattr(agent, "record_search_iteration", lambda *args: None)
    agent.parser = SimpleNamespace(format_tool_result_messages=lambda *args: [])
    buffer = [{"role": "user", "content": "Find a formula."}]
    response_list = [
        ("diagnostic", [], {"role": "assistant", "content": "diagnostic"}),
        ("candidate", [], {"role": "assistant", "content": "candidate"}),
    ]
    results_list = [
        [ToolCallResult(True, {"content": "diagnostic"}, "diagnostic", {})],
        [ToolCallResult(True, {
            "data_split_results": {"train": {"metrics": {"mse": 0.1, "complexity": 2}}},
        }, "candidate", {})],
    ]

    agent.update_buffer(buffer, response_list, results_list, [], {}, list(buffer), {}, R=1, L=1, C=1)

    assert buffer[-2]["content"] == "candidate"


def test_update_topk_records_pareto_front(tmp_path):
    agent = make_agent(tmp_path)
    response_list = [
        ("", [
            ToolCall(name="evaluate_formula", params={"eq": "x"}),
            ToolCall(name="evaluate_formula", params={"eq": "x + y"}),
            ToolCall(name="evaluate_formula", params={"eq": "x + y + z"}),
        ], {}),
    ]
    results_list = [[
        ToolCallResult(True, {
            "formula": "x",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.2, "complexity": 2}},
            },
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.1, "complexity": 3}},
            },
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y + z",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.3, "complexity": 5}},
            },
            "is_candidate": True,
        }, "", {}),
    ]]

    topk_records = agent.update_topk([], response_list, results_list, R=1, L=1, C=1)
    agent.record_search(topk_records, R=1, L=1, C=1)

    lines = (tmp_path / "search_record.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["coord"] == {"R": 1, "C": 1, "L": 1}
    assert [item["formula"] for item in record["pareto_front"]] == ["x + y", "x"]
