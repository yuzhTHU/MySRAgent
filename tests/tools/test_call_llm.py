# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""LLM 工具的单元测试。"""

import pytest
from dotenv import load_dotenv
from sr_agent.tools.call_llm import LLMTool

# 加载环境变量
load_dotenv()


class TestLLMTool:
    """测试 LLMTool 的正确性。"""

    def test_metadata_exists(self):
        """测试元数据存在。"""
        tool = LLMTool()
        assert tool.metadata is not None
        assert tool.metadata.name == "call_llm"

    def test_execute_qwen3_8b(self):
        """测试通过 LLMAPI 工厂调用模型。"""
        class FakeResult:
            usage = {"token": {"prompt": 1}, "price": {"total": 0.0}}

            def __iter__(self):
                yield "2", [], {"role": "assistant", "content": "2"}

        class FakeAPI:
            def __call__(self, messages, n=1):
                assert n == 1
                return FakeResult()

        from sr_agent.api import LLMAPI

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(LLMAPI, "create", lambda *args, **kwargs: FakeAPI())
        tool = LLMTool()
        messages = [{"role": "user", "content": "用一句话回答：1+1 等于几？"}]

        try:
            result = tool.execute(
                llm_provider="siliconflow",
                llm_model="Qwen3-8B",
                messages=messages,
            )
        finally:
            monkeypatch.undo()

        assert result["content"] == "2"
        assert result["token_usage"] == {"prompt": 1}
        assert result["money_usage"] == {"total": 0.0}

    def test_output_structure(self, monkeypatch):
        """测试输出结构完整性。"""
        class FakeResult:
            usage = {"token": {}, "price": {}}

            def __iter__(self):
                yield "Hello", [], {"role": "assistant", "content": "Hello"}

        class FakeAPI:
            def __call__(self, messages, n=1):
                return FakeResult()

        from sr_agent.api import LLMAPI
        monkeypatch.setattr(LLMAPI, "create", lambda *args, **kwargs: FakeAPI())

        tool = LLMTool()
        messages = [{"role": "user", "content": "Hello"}]

        result = tool.execute(
            llm_provider="siliconflow",
            llm_model="Qwen3-8B",
            messages=messages,
        )

        # 检查所有必需的键
        required_keys = ["content", "token_usage", "money_usage"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_execute_invalid_provider(self, monkeypatch):
        """测试无效提供商。"""
        from sr_agent.api import LLMAPI
        monkeypatch.setattr(LLMAPI, "create", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad provider")))

        tool = LLMTool()
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="bad provider"):
            tool.execute(
                llm_provider="invalid_provider",
                llm_model="some-model",
                messages=messages,
            )
