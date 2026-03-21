# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
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


@BaseTool.register('code_executor')
class CodeExecutorTool(BaseTool):
    """Execute given Python code and return printed output.

    This tool provides a restricted Python execution environment that only
    allows importing safe, computation-related modules.

    Use cases:
    - Numerical computation and data processing
    - Mathematical formula verification
    - Algorithm prototyping
    - Expression evaluation in symbolic regression

    Security restrictions:
    - Only whitelisted modules can be imported (numpy, math, random, etc.)
    - File system operations, network access, and system calls are forbidden
    - Dynamic execution functions like eval/exec are forbidden
    - Stdout is captured as output
    """

    metadata = ToolMetadata(
        name="code_executor",
        description="Execute Python code and return printed output. Supports numpy, math libraries for numerical computation and formula verification.",
        category="computation",
    )

    def execute(self, program: str) -> Dict[str, Any]:
        """Execute given Python code.

        Args:
            program: Python code string to execute.

        Returns:
            Dictionary containing:
            - success: Boolean indicating whether execution succeeded
            - output: Captured stdout output
            - error: Error message (if any)
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
        """Validate code for security.

        Args:
            code: Code string to validate.

        Returns:
            Tuple of (is_safe, error_message).
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
        """Create safe global namespace.

        Returns:
            Dictionary containing allowed modules.
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
