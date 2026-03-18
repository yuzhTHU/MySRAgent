"""LLM 工具的单元测试。"""

import pytest
from dotenv import load_dotenv
from sr_agent.tools.llm_tool import LLMTool

# 加载环境变量
load_dotenv()


class TestLLMTool:
    """测试 LLMTool 的正确性。"""

    def test_metadata_exists(self):
        """测试元数据存在。"""
        tool = LLMTool()
        assert tool.metadata is not None
        assert tool.metadata.name == "call_llm"
        assert tool.metadata.category == "llm"

    def test_execute_qwen3_8b(self):
        """测试使用 Qwen3-8B 模型（免费）。"""
        tool = LLMTool()
        messages = [{"role": "user", "content": "用一句话回答：1+1 等于几？"}]

        result = tool.execute(
            llm_provider="siliconflow",
            llm_model="Qwen3-8B",
            messages=messages,
        )

        assert result["success"] is True
        assert result["error"] is None
        assert result["message"] is not None
        assert "1+1" not in result["message"] or "2" in result["message"]
        assert "token_usage" in result
        assert "money_usage" in result

    def test_output_structure(self):
        """测试输出结构完整性。"""
        tool = LLMTool()
        messages = [{"role": "user", "content": "Hello"}]

        result = tool.execute(
            llm_provider="siliconflow",
            llm_model="Qwen3-8B",
            messages=messages,
        )

        # 检查所有必需的键
        required_keys = ["success", "error", "message", "token_usage", "money_usage"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_execute_invalid_provider(self):
        """测试无效提供商。"""
        tool = LLMTool()
        messages = [{"role": "user", "content": "Hello"}]

        result = tool.execute(
            llm_provider="invalid_provider",
            llm_model="some-model",
            messages=messages,
        )

        assert result["success"] is False
        assert result["error"] is not None
