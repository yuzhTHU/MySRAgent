"""代码执行工具。

提供一个安全的 Python 代码执行环境，允许运行无害的计算代码。
"""

import io
import sys
import ast
from typing import Dict, Any, List, Tuple

from .base_tool import BaseTool, ToolMetadata


# 允许的模块白名单
ALLOWED_MODULES = {
    'numpy',
    'math',
    'random',
    'statistics',
    'collections',
    'itertools',
    'functools',
    'operator',
    'decimal',
    'fractions',
    'copy',
    'typing',
    'json',
    're',
    'datetime',
    'time',
    'cmath',
    'array',
    'bisect',
    'heapq',
    'queue',
    'numbers',
}

# 禁止的函数和语句
FORBIDDEN_CALLS = {
    'eval',
    'exec',
    'compile',
    '__import__',
    'open',
    'input',
    'file',
    'reload',
}

# 禁止的模块访问
FORBIDDEN_MODULES = {
    'os',
    'sys',
    'subprocess',
    'socket',
    'urllib',
    'requests',
    'http',
    'ftplib',
    'smtplib',
    'telnetlib',
    'shutil',
    'pathlib',
    'glob',
    'tempfile',
    'pickle',
    'marshal',
    'shelve',
    'sqlite3',
    'csv',
    'configparser',
}


@BaseTool.register_model('code_executor')
class CodeExecutorTool(BaseTool):
    """执行给定的 Python 代码并返回打印输出。

    本工具提供一个受限的 Python 执行环境，只允许导入安全的计算相关模块。
    适用于：
    - 数值计算和数据处理
    - 数学公式验证
    - 算法原型设计
    - 符号回归中的表达式求值

    安全限制：
    - 只能导入白名单中的模块（如 numpy, math, random 等）
    - 禁止文件系统操作、网络访问、系统调用
    - 禁止使用 eval/exec 等动态执行函数
    - 捕获 stdout 作为输出结果
    """

    metadata = ToolMetadata(
        name="code_executor",
        description="执行 Python 代码并返回打印输出。支持 numpy、math 等计算库，适用于数值计算和公式验证。",
        category="computation",
    )

    def execute(self, program: str) -> Dict[str, Any]:
        """执行给定的 Python 代码。

        Args:
            program: 要执行的 Python 代码字符串。

        Returns:
            包含以下字段的字典：
            - success: 布尔值，表示执行是否成功
            - output: 字符串，捕获的 stdout 输出
            - error: 字符串，错误信息（如果有）
        """
        # 验证代码安全性
        is_safe, error_msg = self._validate_code(program)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"代码安全检查失败：{error_msg}"
            }

        # 创建受限的全局命名空间
        global_ns = self._create_safe_globals()
        local_ns = {}

        # 捕获 stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # 执行代码
            exec(program, global_ns, local_ns)
            output = sys.stdout.getvalue()
            return {
                "success": True,
                "output": output,
                "error": ""
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"{type(e).__name__}: {str(e)}"
            }
        finally:
            sys.stdout = old_stdout

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """验证代码是否安全。

        Args:
            code: 要验证的代码字符串。

        Returns:
            (是否安全，错误信息)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误：{e}"

        for node in ast.walk(tree):
            # 检查函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        return False, f"禁止调用函数：{node.func.id}"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN_CALLS:
                        return False, f"禁止调用方法：{node.func.attr}"

            # 检查导入语句
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in FORBIDDEN_MODULES:
                        return False, f"禁止导入模块：{alias.name}"
                    if module_name not in ALLOWED_MODULES:
                        return False, f"未授权的模块：{alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in FORBIDDEN_MODULES:
                        return False, f"禁止导入模块：{node.module}"
                    if module_name not in ALLOWED_MODULES:
                        return False, f"未授权的模块：{node.module}"

        return True, ""

    def _create_safe_globals(self) -> Dict[str, Any]:
        """创建安全的全局命名空间。

        Returns:
            包含允许模块的字典。
        """
        safe_globals = {}

        for module_name in ALLOWED_MODULES:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
                # 添加常用的简短别名
                if module_name == 'numpy':
                    safe_globals['np'] = module
                elif module_name == 'statistics':
                    safe_globals['stat'] = module
                elif module_name == 'collections':
                    safe_globals['collections'] = module
                elif module_name == 'itertools':
                    safe_globals['itertools'] = module
                elif module_name == 'functools':
                    safe_globals['functools'] = module
                elif module_name == 'math':
                    safe_globals['math'] = module
                elif module_name == 'random':
                    safe_globals['random'] = module
            except ImportError:
                pass

        # 添加内置函数（安全的）
        safe_globals.update({
            'print': print,
            'len': len,
            'range': range,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'zip': zip,
            'enumerate': enumerate,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'float': float,
            'int': int,
            'str': str,
            'bool': bool,
            'complex': complex,
        })

        return safe_globals
