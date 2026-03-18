from logging import getLogger
from ..utils.factory import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')


class BaseTool(FactoryMixin):
    """
    Base class for Tools.

    Inherits from FactoryMixin which provides:
    - BaseTool.register_model('name'): Decorator to register subclasses
    - BaseTool.create(...): Factory method to create instances

    The base class is automatically registered as 'default' model.
    Example:
        @BaseTool.register_model('some_tool')
        class SomeTool(BaseTool):
            def __init__(self, ...):
                super().__init__(...)
                # ... custom architecture ...

        # Usage
        tool = BaseToolConfig.create('some_tool', ...)
    """
    pass

    def __init__(self, ...):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method to define tool behavior.")