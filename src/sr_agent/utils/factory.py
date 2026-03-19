# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Dict, Type, TypeVar

T = TypeVar('T', bound='FactoryMixin')


class FactoryMixin:
    REGISTERY_DICT: Dict[str, Type] = {}
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
            cls.REGISTERY_DICT[name] = model_cls
            model_cls.model_name = name
            return model_cls
        return decorator

    @classmethod
    def create(cls: Type[T], name, *args, **kwargs) -> T:
        """
        Factory method to create instance based on the specified name.

        Args:
            name: The name of the model to create
            *args: Positional arguments passed to constructor
            **kwargs: Keyword arguments passed to constructor

        Returns:
            Instantiated object of the type specified in config.model

        Raises:
            ValueError: If config.model is not found in REGISTERY_DICT
        """
        # 'default' refers to the base class itself
        if name == 'default':
            model_class = cls
        elif name in cls.REGISTERY_DICT:
            model_class = cls.REGISTERY_DICT[name]
        else:
            available = ['default'] + list(cls.REGISTERY_DICT.keys())
            raise ValueError(
                f"Unknown model type: '{name}'. "
                f"Available models: {available}. "
                f"Make sure the model class is imported (registration happens at import time)."
            )

        return model_class(*args, **kwargs)
