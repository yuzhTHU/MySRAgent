# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""AskHumanTool 的测试。"""

from sr_agent.tools.ask_human import AskHumanTool


class TestAskHumanTool:
    """测试人类反馈工具。"""

    def test_metadata(self):
        """测试工具元数据。"""
        assert AskHumanTool.metadata.name == "ask_human"
        assert AskHumanTool.metadata.description is not None

    def test_no_callback_returns_default(self):
        """无回调时返回默认消息。"""
        tool = AskHumanTool()
        result = tool.execute("What should I do next?")
        assert "human_response" in result
        assert "No human available" in result["human_response"]

    def test_with_callback(self):
        """有回调时正确调用并返回结果。"""
        responses = []

        def mock_callback(message: str) -> str:
            responses.append(message)
            return "Try sin(x1 + x2)"

        tool = AskHumanTool(human_input_callback=mock_callback)
        result = tool.execute("Best MSE=0.01. What formula structure?")
        assert result["human_response"] == "Try sin(x1 + x2)"
        assert len(responses) == 1
        assert "Best MSE=0.01" in responses[0]

    def test_callback_receives_full_message(self):
        """回调接收到完整的 message 参数。"""
        received = {}

        def mock_callback(message: str) -> str:
            received["msg"] = message
            return "ok"

        tool = AskHumanTool(human_input_callback=mock_callback)
        tool.execute("Summary: found y=2*x. Question: simplify?")
        assert received["msg"] == "Summary: found y=2*x. Question: simplify?"

    def test_format_result_dict(self):
        """测试结果格式化。"""
        result = {"human_response": "Use polynomial features"}
        formatted = AskHumanTool.format_result_dict(result)
        assert "Use polynomial features" in formatted

    def test_tool_call_interface(self):
        """测试通过 __call__ 接口调用。"""
        tool = AskHumanTool(human_input_callback=lambda m: "response")
        tool_result = tool(message="test")
        assert tool_result.ok is True
        assert tool_result.result["human_response"] == "response"
