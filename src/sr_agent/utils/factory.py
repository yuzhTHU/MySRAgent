# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Dict, Type, TypeVar

T = TypeVar('T', bound='FactoryMixin')


class FactoryMixin:
    """
    Mixin class that provides factory pattern capabilities for any class.

    Any class inheriting from this mixin gains the ability to:
    1. Register subclasses via @<Class>.register_model('name')
    2. Create instances via <Class>.create(config, *args, **kwargs)

    The base class is automatically registered as 'default' model.

    ═══════════════════════════════════════════════════════════════════════════
    INHERITANCE ORDER (IMPORTANT)
    ═══════════════════════════════════════════════════════════════════════════

    FactoryMixin should be placed BEFORE other base classes in the inheritance:

        class MyModel(FactoryMixin, nn.Module):
            ...

    This ensures correct Method Resolution Order (MRO) so that:
    1. create() method resolves correctly
    2. Subclasses share the same MODEL_DICT through the mixin

    ═══════════════════════════════════════════════════════════════════════════
    EXAMPLE
    ═══════════════════════════════════════════════════════════════════════════

    ```python
    from nd2py.utils import FactoryMixin

    class MyModel(FactoryMixin, nn.Module):
        def __init__(self, config, tokenizer):
            super().__init__()
            self.config = config

    @MyModel.register_model('gcn')
    class GCNModel(MyModel):
        def __init__(self, config, tokenizer):
            super().__init__(config, tokenizer)
            # ... custom architecture

    # Usage - create() automatically handles model selection
    config = MyConfig(model='gcn')
    model = MyModel.create(config, tokenizer)

    # 'default' is automatically the base class
    config = MyConfig(model='default')
    model = MyModel.create(config, tokenizer)  # Returns MyModel instance
    ```
    """

    MODEL_DICT: Dict[str, Type] = {}
    """Registry of model classes keyed by name. Shared across all subclasses."""

    @classmethod
    def register_model(cls, name: str):
        """
        Decorator to register a model subclass.

        Usage:
            @MyModel.register_model('gcn')
            class GCNModel(MyModel):
                ...
        """
        def decorator(model_cls: type) -> type:
            cls.MODEL_DICT[name] = model_cls
            model_cls.model_name = name
            return model_cls
        return decorator

    @classmethod
    def create(cls: Type[T], config, *args, **kwargs) -> T:
        """
        Factory method to create instance based on config.model.

        Args:
            config: Configuration object with model type specified in config.model
            *args: Positional arguments passed to constructor
            **kwargs: Keyword arguments passed to constructor

        Returns:
            Instantiated object of the type specified in config.model

        Raises:
            ValueError: If config.model is not found in MODEL_DICT
        """
        model_name = getattr(config, 'model', 'default')

        # 'default' refers to the base class itself
        if model_name == 'default':
            model_class = cls
        elif model_name in cls.MODEL_DICT:
            model_class = cls.MODEL_DICT[model_name]
        else:
            available = ['default'] + list(cls.MODEL_DICT.keys())
            raise ValueError(
                f"Unknown model type: '{model_name}'. "
                f"Available models: {available}. "
                f"Make sure the model class is imported (registration happens at import time)."
            )

        return model_class(config, *args, **kwargs)
