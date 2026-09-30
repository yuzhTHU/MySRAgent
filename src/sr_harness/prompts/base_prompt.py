from typing import List, Dict
from logging import getLogger
from ..utils import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')


class BasePrompt(FactoryMixin):
    """
    Base class for Prompts.

    Inherits from FactoryMixin which provides:
    - BasePrompt.register('name'): Decorator to register subclasses
    - BasePrompt.create(...): Factory method to create instances

    The base class is automatically registered as 'default' model.
    Example:
        @BasePrompt.register('some_prompt')
        class SomePrompt(BasePrompt):
            def __init__(self, ...):
                super().__init__(...)
                # ... custom architecture ...

        # Usage
        tool = BasePromptConfig.create('some_prompt', ...)
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwds) -> str | List[Dict[str, str]]:
        """ 从同名文件加载 prompt 模板（如果 prompt 比较短也可以直接从常量字符串加载）"""
        raise NotImplementedError("BasePrompt is an abstract class. Implement __call__ in subclasses.")
