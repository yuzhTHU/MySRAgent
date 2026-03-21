# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""LLM 调用工具。

提供统一的 LLM 调用接口，支持多种 LLM 提供商。
"""

from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('call_llm')
class LLMTool(BaseTool):
    """Call LLM API to generate text responses.

    This tool provides a unified interface for LLM API calls, supporting
    multiple providers (OpenAI, DeepSeek, Gemini, etc.).
    Returns the LLM response content along with token usage and cost information.

    Use cases:
    - Text generation and dialogue
    - Explaining data analysis results
    - Natural language explanations for symbolic regression formulas
    - Any task requiring LLM capabilities

    Supported providers:
    - openai: OpenAI GPT series
    - deepseek: DeepSeek-V3/V2.5
    - gemini: Google Gemini
    - siliconflow: SiliconFlow
    - openrouter: OpenRouter aggregation platform
    - manual: Manual input (for testing)
    """

    metadata = ToolMetadata(
        name="call_llm",
        description="Call LLM API to generate text responses. Supports openai, deepseek, gemini providers. Returns response content, token usage, and cost info.",
        category="llm",
    )

    def execute(
        self,
        llm_provider: str,
        llm_model: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Call LLM API.

        Args:
            llm_provider: LLM provider name, e.g., "openai", "deepseek", "gemini".
            llm_model: Model name, e.g., "gpt-4o-mini", "deepseek-chat".
            messages: List of messages, each as {"role": "user"|"assistant", "content": "..."}.

        Returns:
            Dictionary containing:
            - success: Whether the call succeeded
            - error: Error message (if failed)
            - message: LLM response content
            - token_usage: Token usage statistics (prompt, completion, total)
            - money_usage: Cost statistics (USD)
        """
        try:
            from ..api.llm_api import LLMAPI

            # 实例化 LLM API
            api = LLMAPI.load(llm_provider, llm_model)
            result = api(messages)

            # 获取回复和用量信息
            message = result.contents[0] if result.contents else ""
            usage = result.usage

            return {
                "success": True,
                "error": None,
                "message": message,
                "token_usage": usage.get("token", {}),
                "money_usage": usage.get("price", {}),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": None,
                "token_usage": {},
                "money_usage": {},
            }
