# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from abc import ABCMeta
from typing import Dict, Type, TypeVar

T = TypeVar('T', bound='FactoryMixin')

class FactoryMeta(ABCMeta):
    @property
    def all_registered_classes(cls) -> List[Type]:
        """Return the registry of all registered classes."""
        return list(cls.REGISTRY_DICT.values())
    
    @property
    def all_registered_names(cls) -> List[str]:
        """Return the registry of all registered class names."""
        return list(cls.REGISTRY_DICT.keys())


class FactoryMixin(metaclass=FactoryMeta):
    """
    Mixin class that provides factory pattern capabilities for any class.

    Any class inheriting from this mixin gains the ability to:
    1. Register subclasses via @<Class>.register('name')
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
    2. Each subclass has its own independent registry

    ═══════════════════════════════════════════════════════════════════════════
    EXAMPLE
    ═══════════════════════════════════════════════════════════════════════════

    ```python
    from nd2py.utils import FactoryMixin

    class MyModel(FactoryMixin, nn.Module):
        def __init__(self, config, tokenizer):
            super().__init__()
            self.config = config

    @MyModel.register('gcn')
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

    REGISTRY_DICT: Dict[str, Type] = None  # type: ignore
    """
    Registry of model classes keyed by name.

    Direct subclasses of FactoryMixin get their own registry.
    Grandchildren share the registry with their parent class.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically called when a subclass is created.

        Only create a new registry for direct subclasses of FactoryMixin.
        Grandchildren will share the same registry with their parent,
        so registering on StatsTool also makes it visible to BaseTool.
        """
        super().__init_subclass__(**kwargs)
        # Check if FactoryMixin is a direct parent (not just ancestor)
        if FactoryMixin in cls.__bases__:
            cls.REGISTRY_DICT = {}
        # else: inherit REGISTRY_DICT from parent (already via normal inheritance)

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model subclass.

        Usage:
            @MyModel.register('gcn')
            class GCNModel(MyModel):
                ...
        """
        def decorator(model_cls: type) -> type:
            cls.REGISTRY_DICT[name] = model_cls
            model_cls.model_name = name
            return model_cls
        return decorator

    @classmethod
    def create(cls: Type[T], name: str, *args, create_instance=True, **kwargs) -> T | Type[T]:
        """
        Factory method to create instance based on config.model.

        Args:
            config: Configuration object with model type specified in config.model
            *args: Positional arguments passed to constructor
            create_instance: 
            **kwargs: Keyword arguments passed to constructor

        Returns:
            Instantiated object of the type specified in config.model

        Raises:
            ValueError: If config.model is not found in REGISTRY_DICT
        """
        # 'default' refers to the base class itself
        if name == 'default':
            model_class = cls
        elif name in cls.REGISTRY_DICT:
            model_class = cls.REGISTRY_DICT[name]
        else:
            available = ['default'] + list(cls.REGISTRY_DICT.keys())
            raise ValueError(
                f"Unknown model type: '{name}'. "
                f"Available models: {available}. "
                f"Make sure the model class is imported (registration happens at import time)."
            )

        return model_class(*args, **kwargs) if create_instance else model_class
