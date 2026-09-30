# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Dict, Tuple

__all__ = ["setup_lazy_imports", "TYPE_CHECKING"]

def setup_lazy_imports(module_name: str, import_mapping: Dict[str, Tuple[str, str]]):
    def __getattr__(name: str):
        if name in import_mapping:
            module_path, requires = import_mapping[name]
            try:
                module = importlib.import_module(module_path, package=module_name)
                # 如果这个模块里有同名的属性，返回之
                # 如果没有，说明用户要的就是这个子模块本身，直接返回模块对象
                return getattr(module, name) if hasattr(module, name) else module
            except ImportError as e:
                raise ImportError(
                    f"Failed to import '{name}' from '{module_path}' in module '{module_name}' since missing optional dependency."
                    f"Try to run `pip install nd2py[{requires}]` or `pip install nd2py[all]` to install the required dependencies."
                ) from e
                
        raise AttributeError(f"模块 {module_name!r} 中不存在属性 {name!r}")

    def __dir__():
        return list(import_mapping.keys())

    # 获取调用者的 globals
    caller_globals = sys._getframe(1).f_globals
    
    __all__ = [name for name in caller_globals.keys() if not name.startswith('_')]
    __all__.extend(import_mapping.keys())

    return __getattr__, __dir__, __all__
