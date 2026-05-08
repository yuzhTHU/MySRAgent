# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""LLM 调用工具。提供统一的 LLM 调用接口，支持多种 LLM 提供商。"""

from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata


# @BaseTool.register('call_llm') # 不注册这个工具，因为它用处不大
class LLMTool(BaseTool):
    metadata = ToolMetadata(name="call_llm")

    def execute(
        self,
        llm_provider: str,
        llm_model: str,
        messages: List[Dict[str, str]] | str,
    ) -> Dict[str, Any]:
        """Call LLM API.

        Args:
            llm_provider: LLM provider name, e.g., "openai", "deepseek", "gemini".
            llm_model: Model name, e.g., "gpt-4o-mini", "deepseek-chat".
            messages: List of messages, each as [{"role": "user"|"assistant", "content": "..."}, ...].
        """
        from ..api import LLMAPI

        api = LLMAPI.create(llm_provider, llm_model)
        content = ""
        for content, _, _ in (llm_result := api(messages, n=1)):
            pass
        return {
            "content": content,
            "token_usage": llm_result.usage.get("token", {}),
            "money_usage": llm_result.usage.get("price", {}),
        }
