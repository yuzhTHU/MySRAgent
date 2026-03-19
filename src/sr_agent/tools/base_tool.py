"""工具基类定义。

所有工具都应继承自 BaseTool，并提供统一的接口。
"""

from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from ..utils.factory import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')

@dataclass
class ToolMetadata:
    """工具元数据。

    Attributes:
        name: 工具名称，用于 LLM 识别和调用。
        description: 工具描述，说明工具的功能和适用场景。
        category: 工具类别，用于分类管理（如 "statistics", "regression"）。
    """

    name: str
    description: str
    category: str = "default"


class BaseTool(ABC, FactoryMixin):
    """工具基类。

    所有工具都应继承此类，并实现 execute 方法。
    工具的 docstring 会作为给 LLM 的说明。

    Example:
        @BaseTool.register_model('statistics_tool')
        class StatisticsTool(BaseTool):
            '''计算数据的统计量。'''
            def execute(self, *args, **kwargs):
                return {"mean": np.mean(kwargs.get('y'))}
    """
    def __init__(self, **context):
        """ context 中传入一些工具执行时需要的上下文信息，如数据、模型等，这些信息不适合放在 execute 的参数列表中让 LLM 生成 """
        self.context = context

    metadata: ToolMetadata = None

    @classmethod
    def load_tool_list(cls) -> list[dict]:
        """加载所有已注册工具的元数据列表。

        Returns:
            包含所有工具元数据的列表，每个元素为字典：
            - name: 工具名称
            - description: 工具描述
            - category: 工具类别
            - class_name: 工具类名

        Example:
            >>> tools = BaseTool.load_tool_list()
            >>> for tool in tools:
            ...     print(f"{tool['name']}: {tool['description']}")
        """
        tool_list = []
        for name, tool_cls in cls.MODEL_DICT.items():
            metadata = getattr(tool_cls, 'metadata', None)
            tool_list.append({
                'name': metadata.name if metadata else name,
                'description': metadata.description if metadata else '',
                'category': metadata.category if metadata else 'default',
                'class_name': tool_cls.__name__,
            })
        return tool_list

    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行工具。这个方法的参数列表需要由 LLM 生成，因此其参数应该尽量简单，复杂的上下文信息（如数据）可以通过工具实例的 context 属性传入。

        Args:
            *args: 传递给工具的参数。
            **kwargs: 传递给工具的关键词参数。

        Returns:
            执行结果字典。
        """
        pass

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """允许像函数一样调用工具。"""
        return self.execute(*args, **kwargs)
