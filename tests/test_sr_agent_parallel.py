from __future__ import annotations

from sr_agent.api.core import ToolCall
from sr_agent.sr_agent import SRAgent
from sr_agent.tools.base_tool import BaseTool, ToolMetadata


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
