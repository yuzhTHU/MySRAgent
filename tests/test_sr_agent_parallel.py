from __future__ import annotations

import json

from sr_agent.api.core import ToolCall
from sr_agent.sr_agent import SRAgent
from sr_agent.tools.base_tool import BaseTool, ToolCallResult, ToolMetadata


@BaseTool.register("unit_parallel_tool")
class UnitParallelTool(BaseTool):
    metadata = ToolMetadata(name="unit_parallel_tool")

    def execute(self, value: int) -> dict:
        """Return a value with inherited context."""
        return {"value": value, "offset": self.context["offset"]}


def make_agent(tmp_path):
    agent = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["unit_parallel_tool"],
        save_path=str(tmp_path),
    )
    agent.tools = [UnitParallelTool(offset=10)]
    return agent


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
    assert "At the final refinement round (L = 7)" in system_content
    assert "final-answer mechanism" in system_content


def test_build_prompt_injects_iteration_status_without_mutating_buffer(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_restart_loop = 2
    agent.global_width = 3
    agent.max_refinement_depth = 5
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]

    prompt = agent.build_prompt(buffer, R=1, L=2, C=1)

    assert len(prompt) == len(buffer) + 1
    assert prompt[0]["content"] == "Base system."
    assert "refinement round L=2/5" in prompt[-1]["content"]
    assert "After this response, 3 refinement round(s) remain" in prompt[-1]["content"]
    assert buffer[0]["content"] == "Base system."
    assert buffer[-1]["content"] == "Solve the task."


def test_build_prompt_final_round_tells_agent_to_submit(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 4
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]

    prompt = agent.build_prompt(buffer, R=1, L=4, C=1)
    final_status = prompt[-1]["content"]

    assert "final refinement round" in final_status
    assert "Submit or state your best available target formula now" in final_status
    assert "Do not spend this response on new data exploration" in final_status


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
            "metrics": {"mse": 0.2, "complexity": 2},
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y",
            "metrics": {"mse": 0.1, "complexity": 3},
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y + z",
            "metrics": {"mse": 0.3, "complexity": 5},
            "is_candidate": True,
        }, "", {}),
    ]]

    topk_records = agent.update_topk([], response_list, results_list, R=1, L=1, C=1)
    agent.record_search(topk_records, R=1, L=1, C=1)

    lines = (tmp_path / "search_record.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["coord"] == {"R": 1, "C": 1, "L": 1}
    assert [item["formula"] for item in record["pareto_front"]] == ["x + y", "x"]
