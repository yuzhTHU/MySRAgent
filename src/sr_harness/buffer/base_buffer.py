from typing import List, Dict
from logging import getLogger
from ..utils import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')


class BaseBuffer(FactoryMixin):
    """
    Base class for Buffers.

    Inherits from FactoryMixin which provides:
    - BaseBuffer.register('name'): Decorator to register subclasses
    - BaseBuffer.create(...): Factory method to create instances

    The base class is automatically registered as 'default' model.
    Example:
        @BaseBuffer.register('some_buffer')
        class SomeBuffer(BaseBuffer):
            def __init__(self, ...):
                super().__init__(...)
                # ... custom architecture ...

        # Usage
        tool = BaseBufferConfig.create('some_buffer', ...)
    """

    def __init__(self):
        pass

    def format(self, *args, **kwds) -> List[Dict[str, str]]:
        raise NotImplementedError("BaseBuffer is an abstract class. Implement format in subclasses.")
