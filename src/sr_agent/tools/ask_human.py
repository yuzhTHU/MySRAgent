# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""人类反馈工具。

允许 Agent 暂停执行并向人类请求反馈。
"""
from typing import Any, Dict
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("ask_human")
class AskHumanTool(BaseTool):
    metadata = ToolMetadata(name="ask_human")

    def execute(self, message: str) -> Dict[str, Any]:
        """Pause execution and request human input needed to continue. Use this as a normal 
            collaboration step during planning or execution whenever continuing autonomously 
            would require a meaningful assumption about user intent, scientific goals, constraints, 
            preferences, evaluation criteria, search direction, or a high-impact tradeoff.

        Args:
            message: A message containing progress summary followed by a question.
                The message should first summarize current progress (what we have tried,
                what the best result is so far, and what we have learned), then asks a clear question
                about what direction to explore next. A human expert will read this summary and reply with guidance.
        """
        if (callback := self.context.get("human_input_callback")) is None:
            return {"human_response": "(No human available.)"}
        else:
            return {"human_response": callback(message)}

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        return f"Human response: {result['human_response']}"
