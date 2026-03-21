# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工具调用解析器模块。

负责将工具列表格式化为 LLM 可读的描述，以及从 LLM 响应中解析工具调用。
"""
from __future__ import annotations
from logging import getLogger
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from ..tools import BaseTool
from ..utils import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')


class BaseParser(ABC, FactoryMixin):
    """工具调用解析器基类。

    提供两个核心功能：
    1. format_tools: 将可用的工具列表格式化为 LLM 可读的描述，并通知 LLM 调用工具的格式
    2. parse_response: 根据 format_tools 中确定的格式，鲁棒地从 LLM 响应中解析工具调用

    子类可以为此接口实现不同风格的工具调用和解析格式（如 Text, JSON, XML, Args 等）。
    """

    def __init__(self, tool_list: List[str] | None = None):
        """初始化工具解析器。

        Args:
            tool_list: 可用的工具列表，None 表示使用全部工具。
        """
        if tool_list is not None:
            self.tools = [t for t in BaseTool.load_tool_list() if t['name'] in tool_list]
        else:
            self.tools = BaseTool.load_tool_list()

    @abstractmethod
    def format_tools(self) -> str:
        """将工具列表格式化为 LLM 可读的描述字符串。

        Returns:
            格式化后的工具描述字符串，包含每个工具的名称、描述、签名和使用示例。
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """从 LLM 响应中解析工具调用。

        Args:
            response: LLM 的原始响应文本。

        Returns:
            工具调用列表，每个元素为 (tool_name, params) 元组。
        """
        pass
