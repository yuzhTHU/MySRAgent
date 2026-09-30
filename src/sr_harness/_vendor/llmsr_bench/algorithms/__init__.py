"""
符号回归算法集合
"""

import importlib
from pathlib import Path

# 算法目录
ALGORITHMS_DIR = Path(__file__).parent

def get_algorithm(name: str):
    """获取指定算法的 run 函数"""
    module = importlib.import_module(f".{name}", __name__)
    return getattr(module, "run")


def get_update_parser(name: str):
    """获取指定算法的 update_parser 函数"""
    module = importlib.import_module(f".{name}", __name__)
    return getattr(module, "update_parser", None)


def list_algorithms():
    """列出所有可用的算法"""
    algorithms = []
    for p in ALGORITHMS_DIR.glob("*.py"):
        if p.stem != "__init__":
            algorithms.append(p.stem)
    for p in ALGORITHMS_DIR.iterdir():
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("__"):
            algorithms.append(p.name)
    return sorted(algorithms)
