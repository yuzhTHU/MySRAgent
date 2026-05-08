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
    )
    agent.tools = [UnitParallelTool(offset=10)]
    agent.save_path = str(tmp_path)
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
