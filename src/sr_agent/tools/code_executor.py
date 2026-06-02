# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码执行工具。

提供一个带资源限制的 Python 代码执行环境，允许运行无害的计算代码。
"""

from __future__ import annotations

import io
import os
import ast
import sys
import json
import time
import queue
import signal
import builtins
import traceback
import contextlib
import multiprocessing as mp
from types import ModuleType
from typing import Any, Dict, Tuple
from .base_tool import BaseTool, ToolMetadata


def root_module_name(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def sandbox_module(name: str):
    """Register a factory that creates a restricted module for sandbox imports."""
    def decorator(func):
        func._sandbox_resource_kind = "module"
        func._sandbox_resource_name = name
        return func
    return decorator


def sandbox_builtin(name: str):
    """Register a factory that creates a restricted builtin for sandbox globals."""
    def decorator(func):
        func._sandbox_resource_kind = "builtin"
        func._sandbox_resource_name = name
        return func
    return decorator


class LimitedWriter(io.StringIO):
    """StringIO with a hard byte-ish character cap for untrusted output."""

    SUFFIX = "...[truncated]"

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, text: str) -> int:
        if (remaining := self.limit - self.tell() - len(self.SUFFIX)) <= 0:
            self.truncated = True
            return len(text)
        elif len(text) > remaining:
            super().write(text[:remaining])
            super().write(self.SUFFIX)
            self.truncated = True
            return len(text)
        else:
            return super().write(text)


class SandBoxCodeExecutor:
    # 限制子进程的环境变量，避免外部库（如 numpy）尝试使用多线程导致资源争用和不稳定。
    SAFE_ENVIRONMENT = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    # 允许的模块白名单, 列在此处的模块可以被无条件导入
    # 禁止将可能导致安全问题或资源滥用的模块加入允许列表
    # 如果需要使用特定功能，建议通过 @sandbox_module 注册受限版本
    ALLOWED_MODULES = {
        "array", "bisect", "cmath", "collections", "copy", "datetime",
        "decimal", "fractions", "functools", "heapq", "itertools", "json", 
        "math", "numbers", "numpy", "operator", "queue", "random", 
        "re", "scipy", "statistics", "time", "traceback", "typing",
    }

    # 禁止的内置函数和方法，尤其是那些可能导致安全问题或资源滥用的。
    FORBIDDEN_CALLS = {
        "__import__", "breakpoint", "compile", "delattr", "dir",
        "eval", "exec", "getattr", "globals", "help", "input",
        "locals", "open", "setattr", "vars",
    }

    # 允许 eval() 调用的数学函数和常量，必须通过严格的 AST 验证确保安全。
    MATH_EVAL_FUNCTIONS = {
        "abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh",
        "exp", "fabs", "floor", "log", "log10", "log2", "max", "min",
        "pow", "round", "sin", "sinh", "sqrt", "tan", "tanh",
    }

    # 允许 eval() 调用的数学常量，必须通过严格的 AST 验证确保安全。
    MATH_EVAL_MODULES = {"math", "np", "numpy"}
    MATH_EVAL_CONSTANTS = {"e", "pi", "tau", "inf", "nan"}

    # 允许出现在 eval 表达式中的 AST 节点类型，必须通过严格的 AST 验证确保安全。
    MATH_EVAL_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, 
        ast.Name, ast.Load, ast.Constant, ast.Attribute, 
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, 
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )

    SAFE_BUILTIN_NAMES = {
        "ArithmeticError", "AssertionError", "BaseException", "Exception", "False",
        "FloatingPointError", "IndexError", "KeyError", "KeyboardInterrupt",
        "LookupError", "NameError", "None", "NotImplemented", "OSError", "OverflowError",
        "RuntimeError", "StopIteration", "SyntaxError", "SystemExit", "True",
        "TypeError", "ValueError", "ZeroDivisionError", "__build_class__", 
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "classmethod",
        "complex", "dict", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hash", "hex", "hasattr", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
        "pow", "print", "property", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "zip",
    }

    # 缓存
    _SANDBOX_MODULES: Dict[str, ModuleType] = {}
    _SANDBOX_BUILTINS: Dict[str, Any] = {}

    # ====================
    # 代码安全性检查
    # ====================

    @classmethod
    def validate_code(cls, code: str) -> Dict[str, bool | str]:
        """Validate code before sending it to the sandbox process."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"is_safe": False, "error_msg": f"Code syntax error: {e}"}

        parent_by_child = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        for node in ast.walk(tree):
            ## 函数
            if isinstance(node, ast.Call):
                # eval(...) 只能调用经过严格验证的数学表达式
                if isinstance(node.func, ast.Name) and node.func.id == "eval":
                    if not (validate_result := cls.validate_eval(node))['is_safe']:
                        return {"is_safe": False, "error_msg": validate_result['error_msg']}
                # FORBIDDEN_CALLS 中的函数不许调用
                elif isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    return {"is_safe": False, "error_msg": f"Forbidden function call: {node.func.id}"}
                # FORBIDDEN_CALLS 中的方法不许调用
                if isinstance(node.func, ast.Attribute) and node.func.attr in cls.FORBIDDEN_CALLS:
                    return {"is_safe": False, "error_msg": f"Forbidden method call: {node.func.attr}"}

            ## 名称
            if isinstance(node, ast.Name):
                parent = parent_by_child.get(node)
                # 禁止 eval 以直接调用以外的方式被使用
                is_direct_call = isinstance(parent, ast.Call) and parent.func is node
                if node.id == "eval" and not is_direct_call:
                    return {"is_safe": False, "error_msg": "`eval` can only directly call static string mathematical expressions."}
                # 禁止访问未被授权的双下划线名称
                if node.id.startswith("__") and node.id not in {"__name__"}:
                    return {"is_safe": False, "error_msg": f"Forbidden double-underscore name: {node.id}"}

            ## 属性
            if isinstance(node, ast.Attribute):
                # 禁止访问未被授权的双下划线名称
                if node.attr.startswith("__") and node.attr not in {"__name__"}:
                    return {"is_safe": False, "error_msg": f"Forbidden double-underscore attribute: {node.attr}"}

            ## 导入 (import ...)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not (validate_result := cls.validate_import(alias.name))['is_safe']:
                        return {"is_safe": False, "error_msg": validate_result['error_msg']}

            ## 导入 (from ... import ...)
            if isinstance(node, ast.ImportFrom):
                # 禁止相对导入
                if node.level != 0:
                    return {"is_safe": False, "error_msg": "Relative imports are not allowed."}
                if node.module:
                    if not (validate_result := cls.validate_import(node.module))['is_safe']:
                        return {"is_safe": False, "error_msg": validate_result['error_msg']}

        return {"is_safe": True, "error_msg": ""}

    @classmethod
    def validate_eval(cls, node: ast.Call) -> Dict[str, bool | str]:
        """验证 node 是否是一个安全的 eval 语句 (纯数学表达式)"""
        if not node.args or len(node.args) > 3:
            return {"is_safe": False, "error_msg": "Illegal eval call: must have 1-3 positional arguments"}
        expr_arg = node.args[0]
        if not (isinstance(expr_arg, ast.Constant) and isinstance(expr_arg.value, str)):
            # 动态字符串 (eval(variable)) 的安全性不容易确定，交给运行时 safe_eval 处理。
            return {"is_safe": True, "error_msg": ""}
        if not (validate_result := cls.validate_math_expression(expr_arg.value))['is_safe']:
            return {"is_safe": False, "error_msg": validate_result['error_msg']}
        else:
            return {"is_safe": True, "error_msg": ""}

    @classmethod
    def validate_math_expression(cls, expression: str) -> Dict[str, bool | str]:
        """验证 expression 是否是一个安全的纯数学表达式"""
        try:
            expr_tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return {"is_safe": False, "error_msg": f"Expression syntax error: {e}"}

        parent_by_child = {
            child: parent
            for parent in ast.walk(expr_tree)
            for child in ast.iter_child_nodes(parent)
        }

        for node in ast.walk(expr_tree):
            # 禁止使用不在 MATH_EVAL_NODES 中的 AST 节点类型
            if not isinstance(node, cls.MATH_EVAL_NODES):
                return {"is_safe": False, "error_msg": f"Forbidden expression node: {type(node).__name__}"}

            # 禁止调用不在 MATH_EVAL_FUNCTIONS 中的函数
            if isinstance(node, ast.Call):
                if node.keywords:
                    return {"is_safe": False, "error_msg": "Forbidden keyword arguments in eval math expression"}
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    return {"is_safe": False, "error_msg": "Forbidden dynamic function call"}
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec"}:
                        return {"is_safe": False, "error_msg": f"Forbidden nested call: {node.func.id}"}
                    if node.func.id not in cls.MATH_EVAL_FUNCTIONS:
                        return {"is_safe": False, "error_msg": f"Forbidden function call: {node.func.id}"}
                if isinstance(node.func, ast.Attribute):
                    if not (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in cls.MATH_EVAL_MODULES
                        and node.func.attr in cls.MATH_EVAL_FUNCTIONS
                    ):
                        return {"is_safe": False, "error_msg": f"Forbidden method call: {ast.unparse(node.func)}"}

            # 禁止使用非数字类型的常量
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, complex)):
                    return {"is_safe": False, "error_msg": f"Forbidden constant type: {type(node.value).__name__}"}

            # 禁止访问未被授权的名称
            if isinstance(node, ast.Name):
                if node.id in {"eval", "exec"}:
                    return {"is_safe": False, "error_msg": f"Forbidden nested call: {node.id}"}

                if node.id.startswith("__"):
                    return {"is_safe": False, "error_msg": f"Forbidden double underscore name: {node.id}"}

                if not (
                    isinstance(parent := parent_by_child.get(node), ast.Attribute)
                    and parent.value is node
                    and node.id in cls.MATH_EVAL_MODULES
                ):
                    if node.id in (cls.MATH_EVAL_MODULES | cls.ALLOWED_MODULES):
                        return {"is_safe": False, "error_msg": f"Forbidden direct module access: {node.id}"}
                    if node.id in cls.SAFE_BUILTIN_NAMES and node.id not in (cls.MATH_EVAL_FUNCTIONS | cls.MATH_EVAL_CONSTANTS):
                        return {"is_safe": False, "error_msg": f"Forbidden non-mathematical built-in name: {node.id}"}

            # 禁止访问未被授权的属性
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    return {"is_safe": False, "error_msg": f"Forbidden double underscore attribute: {node.attr}"}
                if not (
                    isinstance(node.value, ast.Name)
                    and node.value.id in cls.MATH_EVAL_MODULES
                    and node.attr in (cls.MATH_EVAL_FUNCTIONS | cls.MATH_EVAL_CONSTANTS)
                ):
                    return {"is_safe": False, "error_msg": f"Forbidden attribute access: {ast.unparse(node)}"}

        return {"is_safe": True, "error_msg": ""}

    @classmethod
    def validate_import(cls, module_name: str) -> Tuple[bool, str]:
        root = root_module_name(module_name)
        if root in set(cls.iter_sandbox_resource_factories("module")):
            return {"is_safe": True, "error_msg": ""}
        if root not in cls.ALLOWED_MODULES:
            return {"is_safe": False, "error_msg": f"Unauthorized module: {module_name}"}
        return {"is_safe": True, "error_msg": ""}

    # ====================
    # 代码受限执行
    # ====================

    @classmethod
    def sandbox_worker(
        cls,
        program: str,
        stdin_text: str,
        timeout_seconds: int,
        memory_limit_mb: int,
        output_limit_bytes: int,
        result_queue: mp.Queue,
    ) -> None:
        # 准备受限环境
        cls.prepare_sandbox_runtime(
            stdin_text=stdin_text,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
        # 执行代码并捕获输出
        stdout = LimitedWriter(output_limit_bytes)
        stderr = LimitedWriter(output_limit_bytes)
        try:
            code = compile(program, "<llm-generated-code>", "exec")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                safe_builtins = {name: getattr(builtins, name) for name in SandBoxCodeExecutor.SAFE_BUILTIN_NAMES}
                safe_builtins["__import__"] = cls.limited_import
                safe_builtins.update(cls._SANDBOX_BUILTINS)
                safe_globals = {"__builtins__": safe_builtins, "__name__": "__sandbox__", "__package__": None}
                exec(code, safe_globals, safe_globals)
            sandbox_error = None
        except BaseException as e:
            sandbox_error = f"{type(e).__name__}: {e}"
            stderr.write(traceback.format_exc(limit=8))
        result_queue.put({
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "sandbox_error": sandbox_error,
        })

    @classmethod
    def prepare_sandbox_runtime(cls, **sandbox_context) -> None:
        cls._SANDBOX_MODULES = {
            name: factory(**sandbox_context)
            for name, factory in cls.iter_sandbox_resource_factories("module").items()
        }
        cls._SANDBOX_BUILTINS = {
            name: factory(**sandbox_context)
            for name, factory in cls.iter_sandbox_resource_factories("builtin").items()
        }
        cls.apply_resource_limits(sandbox_context["timeout_seconds"], sandbox_context["memory_limit_mb"])
        for key, value in cls.SAFE_ENVIRONMENT.items():
            os.environ[key] = value
        cls.install_sandbox_resources()

    @classmethod
    def install_sandbox_resources(cls) -> None:
        for resource in [*cls._SANDBOX_MODULES.values(), *cls._SANDBOX_BUILTINS.values()]:
            installer = getattr(resource, "_sandbox_install", None)
            if installer is not None:
                installer()

    @classmethod
    def apply_resource_limits(cls, timeout_seconds: int, memory_limit_mb: int) -> None:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as statm_file:
                pages = int(statm_file.read().split()[0])
            current_address_space = pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            current_address_space = 0

        try:
            import resource

            requested_memory_bytes = memory_limit_mb * 1024 * 1024
            memory_bytes = max(
                requested_memory_bytes,
                current_address_space + 256 * 1024 * 1024,
            )
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (memory_bytes, memory_bytes))
            if sys.platform != "darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (memory_bytes, memory_bytes))

            cpu_seconds = max(1, int(timeout_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except Exception:
            # resource 在部分平台不可用；父进程仍会通过 wall timeout 终止子进程。
            pass

    @classmethod
    def limited_import(
        cls,
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ) -> ModuleType:
        """限制子进程的 import 行为，禁止导入未授权的模块。"""
        # 禁止 from .xxx import yyy 这样的相对导入
        if level != 0:
            raise ImportError("relative imports are not allowed")
        # 允许 allowed_module 中的模块进行任意导入
        elif (caller_root := root_module_name((globals or {}).get("__name__", ""))) in cls.ALLOWED_MODULES:
            return builtins.__import__(name, globals, locals, fromlist, level)
        # 允许从 sandbox_module 注册的模块中导入
        elif (root := root_module_name(name)) in cls._SANDBOX_MODULES:
            return cls.restricted_module_for_import(root, cls._SANDBOX_MODULES[root], fromlist)
        # 允许从 ALLOWED_MODULES 中导入
        elif root in cls.ALLOWED_MODULES:
            return builtins.__import__(name, globals, locals, fromlist, level)
        # 禁止从其他未明确允许的模块中导入
        else:
            raise ImportError(f"Unauthorized module: {name}")

    @classmethod
    def restricted_module_for_import(cls, name: str, module: ModuleType, fromlist=()) -> ModuleType:
        for attr in fromlist or ():
            if attr == "*":
                continue
            if not hasattr(module, attr):
                raise ImportError(f"cannot import name '{attr}' from '{name}'")
        return module

    # ====================
    # 沙箱资源
    # ====================

    @classmethod
    def iter_sandbox_resource_factories(cls, kind: str):
        """ 查找所有被 @sandbox_module 或 @sandbox_builtin 装饰的工厂函数，返回一个 name->factory 的字典 """
        factories = {}
        # reversed 是为了让子类定义的资源优先于父类同名资源
        for base in reversed(cls.__mro__):
            for attr_name, attr in base.__dict__.items():
                func = attr.__func__ if isinstance(attr, classmethod) else attr
                if getattr(func, "_sandbox_resource_kind", None) == kind:
                    name = getattr(func, "_sandbox_resource_name")
                    factories[name] = getattr(cls, attr_name)
        return factories

    @classmethod
    @sandbox_builtin("eval")
    def create_safe_eval(cls, **sandbox_context) -> Any:
        """只能执行纯数学表达式的安全 eval 实现，必须通过 validate_math_expression 的 AST 验证才能调用"""
        def safe_eval(expression: str, globals=None, locals=None) -> Any:
            if not isinstance(expression, str):
                raise TypeError("Expression for eval must be a string.")
            if not (validate_result := cls.validate_math_expression(expression))['is_safe']:
                raise ValueError(f"Unsafe expression for eval: {validate_result['error_msg']}")
            env = {}
            if globals is None and locals is None:
                frame = sys._getframe(1)
                globals = frame.f_globals
                locals = frame.f_locals
            if globals is not None:
                env.update(globals)
            if locals is not None:
                env.update(locals)
            return builtins.eval(compile(expression, "<safe-math-eval>", "eval"), {"__builtins__": {}}, env)
        return safe_eval

    @classmethod
    @sandbox_module("sys")
    def create_fake_sys_module(cls, stdin_text: str = "", **sandbox_context) -> ModuleType:
        """只能访问有限属性的 sys 模块，stdin 可通过参数传入，stdout 和 stderr 定向到父进程的 stdout 和 stderr"""
        fake_sys = ModuleType("sys")
        fake_sys.stdin = io.StringIO(stdin_text)
        fake_sys.stdout = sys.stdout
        fake_sys.stderr = sys.stderr
        fake_sys.version = sys.version
        fake_sys.version_info = sys.version_info
        fake_sys.platform = sys.platform
        return fake_sys


@BaseTool.register("code_executor")
class CodeExecutorTool(BaseTool):
    metadata = ToolMetadata(name="code_executor")

    # 计算资源限制
    DEFAULT_TIMEOUT_SECONDS = 30
    DEFAULT_MEMORY_LIMIT_MB = 1024
    DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024 # 64 KB, 折合 token 约为 32K
    MAX_TIMEOUT_SECONDS = 120
    MAX_MEMORY_LIMIT_MB = 4096
    MAX_OUTPUT_LIMIT_BYTES = 1024 * 1024 # 1 MB, 折合 token 约为 512K

    def execute(
        self,
        program: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> Dict[str, Any]:
        """Execute Python code and return printed output.
        1) Use `import sys, json; data_dict = json.loads(sys.stdin.read())` to access data mapping variable names to lists.
        2) Use `print()` to produce output.
        3) The code is executed in a restricted sandbox with resource limits.
        4) numpy and scipy are available, but libraries like matplotlib, pandas, scikit-learn are forbidden.
        5) exec() is forbidden. eval() is allowed only for math expression strings that
           pass a strict AST whitelist; otherwise define and call your own functions.

        Args:
            program: Python code string to execute.
            timeout_seconds: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.
            output_limit_bytes: Limit on the amount of output (in bytes) that can be produced.
        """
        # 准备 stdin
        if not hasattr(self, 'stdin_text'):
            assert 'data' in self.context
            data = self.context['data']
            data_dict = self.serialization(data)
            stdin_text = self.stdin_text = json.dumps(data_dict, ensure_ascii=False)
        else:
            stdin_text = self.stdin_text

        # 准备 program
        program = self.extract_code(program)
        if not (validation_result := SandBoxCodeExecutor.validate_code(program))['is_safe']:
            raise Exception(f"Code security check failed since: {validation_result['error_msg']}")

        # 准备子进程
        timeout_seconds = self.bounded_int(timeout_seconds, self.DEFAULT_TIMEOUT_SECONDS, 1, self.MAX_TIMEOUT_SECONDS)
        memory_limit_mb = self.bounded_int(memory_limit_mb, self.DEFAULT_MEMORY_LIMIT_MB, 64, self.MAX_MEMORY_LIMIT_MB)
        output_limit_bytes = self.bounded_int(output_limit_bytes, self.DEFAULT_OUTPUT_LIMIT_BYTES, 1024, self.MAX_OUTPUT_LIMIT_BYTES)
        mp_context = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")  # Windows (nt) 不支持 fork, 必须用 spawn
        result_queue = mp_context.Queue(maxsize=1)
        worker_args = (program, stdin_text, timeout_seconds, memory_limit_mb, output_limit_bytes, result_queue)
        process = mp_context.Process(target=SandBoxCodeExecutor.sandbox_worker, args=worker_args)
        
        # 启动子进程
        max_retry = 3
        for attempt in range(1, max_retry + 1):
            try:
                process.start()
                break
            except RuntimeError as e:
                if not ("can't start new thread" in str(e) or "Resource temporarily unavailable" in str(e)):
                    raise
                if attempt < max_retry:
                    time.sleep(1.0 * attempt)
        else:
            raise Exception(f"Cannot start sandbox subprocess after {max_retry} attempts: [{type(e).__name__}] {e}")

        # 等待子进程
        result = None
        start_time = time.monotonic()
        deadline = start_time + timeout_seconds
        while time.monotonic() < deadline:
            try:
                result = result_queue.get(timeout=0.05)
                break
            except queue.Empty:
                if not process.is_alive():
                    break
        
        # 处理结果
        if result is None:
            if process.is_alive(): # 超时
                self.terminate_process(process)
                raise Exception(f"Sandbox subprocess did not return result before timeout={timeout_seconds} seconds and has been terminated.")
            elif (exit_code := process.exitcode) is not None and exit_code < 0: # 报错
                raise Exception(f"Sandbox subprocess was terminated by signal {signal.Signals(-exit_code).name}.")
            else: # 无结果
                raise Exception(f"Sandbox subprocess did not return result and has exited with code {exit_code}.")
        elif result["sandbox_error"] is not None:
            raise Exception(f"Sandbox execution error: {result['sandbox_error']}")
        else:
            process.join(1)
            self.terminate_process(process)
            return {
                'stdout': result['stdout'], 
                'stderr': result['stderr'], 
                'duration': time.monotonic() - start_time,
                "timeout_seconds": timeout_seconds,
                "memory_limit_mb": memory_limit_mb,
                "output_limit_bytes": output_limit_bytes,
            }

    @classmethod
    def extract_code(cls, code: str) -> str:
        raw_code = str(code).strip()
        if "```python" in raw_code:
            return raw_code.split("```python")[-1].split("```")[0].strip()
        if "```" not in raw_code:
            return raw_code
        if len(parts := raw_code.split("```")) <= 1:
            return raw_code
        if "\n" not in (potential_code := parts[1]):
            return potential_code.strip()
        first_line, rest_of_code = potential_code.split("\n", 1)
        if first_line.strip().isalpha():
            return rest_of_code.strip()
        return potential_code.strip()

    @classmethod
    def serialization(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls.serialization(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls.serialization(item) for item in value]
        if hasattr(value, "tolist"):
            return cls.serialization(value.tolist())
        if hasattr(value, "item"):
            return cls.serialization(value.item())
        raise TypeError(f"unsupported data type {type(value).__name__}")

    @classmethod
    def terminate_process(cls, process: mp.Process) -> None:
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join()

    @classmethod
    def bounded_int(cls, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))
