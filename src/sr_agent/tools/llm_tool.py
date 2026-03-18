"""LLM 调用工具。

提供统一的 LLM 调用接口，支持多种 LLM 提供商。
"""

from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata


class LLMTool(BaseTool):
    """调用 LLM API 生成文本回复。

    本工具提供统一的 LLM 调用接口，支持多种 LLM 提供商（OpenAI、DeepSeek、Gemini 等）。
    返回 LLM 的回复内容，以及 token 使用和费用信息。

    适用场景：
    - 文本生成和对话
    - 数据分析结果的解释
    - 符号回归公式的自然语言解释
    - 任意需要 LLM 能力的任务

    支持的提供商：
    - openai: OpenAI GPT 系列
    - deepseek: DeepSeek-V3/V2.5
    - gemini: Google Gemini
    - siliconflow: 硅基流动
    - openrouter: OpenRouter 聚合平台
    - manual: 手动输入（测试用）
    """

    metadata = ToolMetadata(
        name="call_llm",
        description="调用 LLM API 生成文本回复。支持 openai、deepseek、gemini 等提供商。返回回复内容、token 使用和费用信息。",
        category="llm",
    )

    def execute(
        self,
        llm_provider: str,
        llm_model: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """调用 LLM API。

        Args:
            llm_provider: LLM 提供商名称，如 "openai"、"deepseek"、"gemini" 等。
            llm_model: 模型名称，如 "gpt-4o-mini"、"deepseek-chat" 等。
            messages: 消息列表，每个消息为 {"role": "user"|"assistant", "content": "..."} 格式。

        Returns:
            包含以下字段的字典：
            - success: 是否成功调用
            - error: 错误信息（如果失败）
            - message: LLM 回复内容
            - token_usage: Token 使用统计（prompt、completion、total 等）
            - money_usage: 费用统计（USD）
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
