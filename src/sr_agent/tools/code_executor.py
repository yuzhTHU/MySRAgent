# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码执行工具。

提供一个带资源限制的 Python 代码执行环境，允许运行无害的计算代码。
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Dict, Tuple

from .base_tool import BaseTool, ToolMetadata


DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024
MAX_TIMEOUT_SECONDS = 120
MAX_MEMORY_LIMIT_MB = 4096
SANDBOX_INPUT_CONTEXT_KEYS = ("stdin_data", "sandbox_data", "input_data", "data")

SAFE_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# 允许的模块白名单。保持偏计算用途，避免文件系统、网络和进程控制能力。
ALLOWED_MODULES = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "numbers",
    "numpy",
    "operator",
    "queue",
    "random",
    "re",
    "scipy",
    "statistics",
    "time",
    "traceback",
    "typing",
}

FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

MATH_EVAL_FUNCTIONS = {
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "fabs",
    "floor",
    "log",
    "log10",
    "log2",
    "max",
    "min",
    "pow",
    "round",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}

MATH_EVAL_MODULES = {"math", "np", "numpy"}

MATH_EVAL_CONSTANTS = {"e", "pi", "tau", "inf", "nan"}

FORBIDDEN_MODULES = {
    "configparser",
    "ctypes",
    "csv",
    "ftplib",
    "glob",
    "http",
    "importlib",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shelve",
    "shutil",
    "signal",
    "smtplib",
    "socket",
    "sqlite3",
    "subprocess",
    "sys",
    "telnetlib",
    "tempfile",
    "threading",
    "urllib",
}

SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "AssertionError",
    "BaseException",
    "Exception",
    "False",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "NameError",
    "None",
    "NotImplemented",
    "OSError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "SyntaxError",
    "SystemExit",
    "True",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "__build_class__",
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "hasattr",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
}


class _LimitedWriter(io.StringIO):
    """StringIO with a hard byte-ish character cap for untrusted output."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, text: str) -> int:
        current = self.tell()
        remaining = self.limit - current
        if remaining <= 0:
            self.truncated = True
            return len(text)
        if len(text) > remaining:
            super().write(text[:remaining])
            self.truncated = True
            return len(text)
        return super().write(text)


def _root_module_name(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _validate_math_eval_expression_source(expression: str) -> Tuple[bool, str]:
    try:
        expr_tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return False, f"eval 数学表达式语法错误：{e}"
    return _validate_math_eval_expression_tree(expr_tree)


def _validate_math_eval_expression_tree(tree: ast.AST) -> Tuple[bool, str]:
    parent_by_child = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False, f"禁止的表达式节点：{type(node).__name__}"

        if isinstance(node, ast.Call):
            is_allowed, error_msg = _validate_math_eval_expression_call(node)
            if not is_allowed:
                return False, error_msg

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex)):
                return False, f"禁止的常量类型：{type(node.value).__name__}"

        if isinstance(node, ast.Name):
            if node.id in {"eval", "exec"}:
                return False, f"禁止嵌套调用：{node.id}"
            if node.id.startswith("__"):
                return False, f"禁止访问双下划线名称：{node.id}"
            parent = parent_by_child.get(node)
            is_module_attr_base = (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and node.id in MATH_EVAL_MODULES
            )
            if is_module_attr_base:
                continue
            if node.id in MATH_EVAL_MODULES | ALLOWED_MODULES | FORBIDDEN_MODULES:
                return False, f"禁止直接访问模块：{node.id}"
            if node.id in SAFE_BUILTIN_NAMES and node.id not in MATH_EVAL_FUNCTIONS | MATH_EVAL_CONSTANTS:
                return False, f"禁止访问非数学内置名称：{node.id}"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"禁止访问双下划线属性：{node.attr}"
            if not (
                isinstance(node.value, ast.Name)
                and node.value.id in MATH_EVAL_MODULES
                and node.attr in MATH_EVAL_FUNCTIONS | MATH_EVAL_CONSTANTS
            ):
                return False, f"禁止访问属性：{ast.unparse(node)}"

    return True, ""


def _validate_math_eval_expression_call(node: ast.Call) -> Tuple[bool, str]:
    if node.keywords:
        return False, "禁止 eval 数学表达式中的关键字参数"

    if isinstance(node.func, ast.Name):
        if node.func.id in {"eval", "exec"}:
            return False, f"禁止嵌套调用：{node.func.id}"
        if node.func.id not in MATH_EVAL_FUNCTIONS:
            return False, f"禁止调用函数：{node.func.id}"
        return True, ""

    if isinstance(node.func, ast.Attribute):
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in MATH_EVAL_MODULES
            and node.func.attr in MATH_EVAL_FUNCTIONS
        ):
            return True, ""
        return False, f"禁止调用方法：{ast.unparse(node.func)}"

    return False, "禁止动态函数调用"


def _safe_math_eval(expression: str, globals=None, locals=None) -> Any:
    if not isinstance(expression, str):
        raise TypeError("eval 只能用于字符串数学表达式")
    is_allowed, error_msg = _validate_math_eval_expression_source(expression)
    if not is_allowed:
        raise ValueError(f"eval 数学表达式不安全：{error_msg}")

    env: Dict[str, Any] = {}
    if globals is None and locals is None:
        frame = sys._getframe(1)
        env.update(frame.f_globals)
        env.update(frame.f_locals)
    else:
        if globals:
            env.update(globals)
        if locals:
            env.update(locals)

    return builtins.eval(compile(expression, "<safe-math-eval>", "eval"), {"__builtins__": {}}, env)


_SANDBOX_STDIN_TEXT = ""
_SANDBOX_FAKE_SYS: ModuleType | None = None


def _create_fake_sys_module(stdin_text: str) -> ModuleType:
    fake_sys = ModuleType("sys")
    fake_sys.stdin = io.StringIO(stdin_text)
    fake_sys.stdout = sys.stdout
    fake_sys.stderr = sys.stderr
    fake_sys.version = sys.version
    fake_sys.version_info = sys.version_info
    fake_sys.platform = sys.platform
    return fake_sys


def _limited_import(
    name: str,
    globals=None,
    locals=None,
    fromlist=(),
    level: int = 0,
) -> ModuleType:
    if level != 0:
        raise ImportError("relative imports are not allowed")
    if name == "sys":
        return _SANDBOX_FAKE_SYS or _create_fake_sys_module(_SANDBOX_STDIN_TEXT)

    caller_name = (globals or {}).get("__name__", "")
    caller_root = _root_module_name(caller_name)
    if caller_root in ALLOWED_MODULES:
        return builtins.__import__(name, globals, locals, fromlist, level)

    root = _root_module_name(name)
    if root in FORBIDDEN_MODULES:
        raise ImportError(f"禁止导入模块：{name}")
    if root not in ALLOWED_MODULES:
        raise ImportError(f"未授权的模块：{name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _apply_resource_limits(timeout_seconds: int, memory_limit_mb: int) -> None:
    try:
        import resource

        requested_memory_bytes = memory_limit_mb * 1024 * 1024
        current_address_space = _current_address_space_bytes()
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


def _current_address_space_bytes() -> int:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as statm_file:
            pages = int(statm_file.read().split()[0])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _create_safe_globals() -> Dict[str, Any]:
    safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe_builtins["__import__"] = _limited_import
    safe_builtins["eval"] = _safe_math_eval
    return {
        "__builtins__": safe_builtins,
        "__name__": "__sandbox__",
        "__package__": None,
    }


def _sandbox_worker(
    program: str,
    stdin_text: str,
    timeout_seconds: int,
    memory_limit_mb: int,
    output_limit_bytes: int,
    result_queue: mp.Queue,
) -> None:
    global _SANDBOX_FAKE_SYS, _SANDBOX_STDIN_TEXT

    for key, value in SAFE_ENVIRONMENT.items():
        os.environ[key] = value

    _SANDBOX_STDIN_TEXT = stdin_text
    _SANDBOX_FAKE_SYS = _create_fake_sys_module(stdin_text)
    _apply_resource_limits(timeout_seconds, memory_limit_mb)

    stdout = _LimitedWriter(output_limit_bytes)
    stderr = _LimitedWriter(output_limit_bytes)
    started_at = time.monotonic()

    try:
        code = compile(program, "<llm-generated-code>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            safe_globals = _create_safe_globals()
            exec(code, safe_globals, safe_globals)
        status = "success"
        error = ""
        exit_code = 0
    except BaseException as exc:
        status = "runtime_error"
        error = f"{type(exc).__name__}: {exc}"
        stderr.write(traceback.format_exc(limit=8))
        exit_code = 1

    result_queue.put(
        {
            "status": status,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": error,
            "exit_code": exit_code,
            "duration": time.monotonic() - started_at,
            "output_truncated": stdout.truncated or stderr.truncated,
        }
    )


def _get_sandbox_multiprocessing_context() -> mp.context.BaseContext:
    """Return a stdlib multiprocessing context for the sandbox worker.

    Joblib's loky backend mutates multiprocessing's default context inside
    worker processes. Using explicit stdlib contexts keeps code_executor's
    nested sandbox process independent from loky internals.
    """
    if os.name == "nt":
        return mp.get_context("spawn")
    return mp.get_context("fork")


@BaseTool.register("code_executor")
class CodeExecutorTool(BaseTool):
    metadata = ToolMetadata(name="code_executor")

    def execute(
        self,
        program: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
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
            timeout: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.

        Returns:
            Dictionary containing:
            - success: Boolean indicating whether execution succeeded
            - output: Captured stdout output
            - error: Error message (if any)
            - status: Sandbox status such as success, security_error, timeout, runtime_error
            - stderr: Captured stderr output
            - exit_code: Process exit code (if available)
            - duration: Runtime in seconds
        """
        timeout = self._bounded_int(timeout, DEFAULT_TIMEOUT_SECONDS, 1, MAX_TIMEOUT_SECONDS)
        memory_limit_mb = self._bounded_int(memory_limit_mb, DEFAULT_MEMORY_LIMIT_MB, 64, MAX_MEMORY_LIMIT_MB)

        program = self._extract_code(program)
        stdin_text, has_stdin_data, input_error = self._get_sandbox_input_text()
        if input_error:
            return self._failure("input_error", input_error)

        is_safe, error_msg = self._validate_code(program)
        if not is_safe:
            return self._failure("security_error", f"代码安全检查失败：{error_msg}")

        mp_context = _get_sandbox_multiprocessing_context()
        result_queue: mp.Queue = mp_context.Queue(maxsize=1)
        process = mp_context.Process(
            target=_sandbox_worker,
            args=(
                program,
                stdin_text,
                timeout,
                memory_limit_mb,
                DEFAULT_OUTPUT_LIMIT_BYTES,
                result_queue,
            ),
        )
        started_at = time.monotonic()
        process.start()

        result = None
        deadline = started_at + timeout
        while time.monotonic() < deadline:
            try:
                result = result_queue.get(timeout=0.05)
                break
            except queue.Empty:
                if not process.is_alive():
                    break

        if result is not None:
            process.join(1)
            if process.is_alive():
                self._terminate_process(process)
        elif process.is_alive():
            self._terminate_process(process)
            return self._failure(
                "timeout",
                f"代码执行超时：超过 {timeout} 秒",
                exit_code=process.exitcode,
                duration=time.monotonic() - started_at,
            )
        else:
            exit_code = process.exitcode
            if exit_code is not None and exit_code < 0:
                signal_name = signal.Signals(-exit_code).name
                return self._failure(
                    "runtime_error",
                    f"子进程被信号终止：{signal_name}",
                    exit_code=exit_code,
                    duration=time.monotonic() - started_at,
                )
            return self._failure(
                "sandbox_error",
                "沙盒进程没有返回执行结果",
                exit_code=exit_code,
                duration=time.monotonic() - started_at,
            )

        success = result["status"] == "success"
        error = result["error"]
        if result["output_truncated"]:
            suffix = "\n[output truncated]"
            if len(result["stdout"]) >= DEFAULT_OUTPUT_LIMIT_BYTES:
                result["stdout"] += suffix
            else:
                result["stderr"] += suffix
            if not error:
                error = "输出超过限制，已截断"

        return {
            "success": success,
            "output": result["stdout"] if success else "",
            "error": error,
            "status": result["status"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
            "duration": result["duration"],
            "timeout": timeout,
            "memory_limit_mb": memory_limit_mb,
            "stdin_data": has_stdin_data,
        }

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code before sending it to the sandbox process."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误：{e}"

        parent_by_child = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "eval":
                    is_allowed, error_msg = self._validate_math_eval_call(node)
                    if not is_allowed:
                        return False, error_msg
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                    return False, f"禁止调用函数：{node.func.id}"
                if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                    return False, f"禁止调用方法：{node.func.attr}"

            if isinstance(node, ast.Name):
                parent = parent_by_child.get(node)
                is_direct_eval_call = (
                    node.id == "eval"
                    and isinstance(parent, ast.Call)
                    and parent.func is node
                )
                if node.id == "eval" and not is_direct_eval_call:
                    return False, "eval 只能直接调用静态字符串数学表达式"
                if node.id.startswith("__") and node.id not in {"__name__"}:
                    return False, f"禁止访问双下划线名称：{node.id}"

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr not in {"__name__"}:
                    return False, f"禁止访问双下划线属性：{node.attr}"

            if isinstance(node, ast.Import):
                for alias in node.names:
                    is_allowed, error_msg = self._validate_import(alias.name)
                    if not is_allowed:
                        return False, error_msg

            if isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    return False, "禁止相对导入"
                if node.module:
                    is_allowed, error_msg = self._validate_import(node.module)
                    if not is_allowed:
                        return False, error_msg

        return True, ""

    def _validate_math_eval_call(self, node: ast.Call) -> Tuple[bool, str]:
        """Allow direct eval(); runtime wrapper validates dynamic math strings."""
        if not node.args:
            return False, "禁止调用函数：eval"
        if len(node.args) > 3:
            return False, "eval 只能用于数学表达式"

        expr_arg = node.args[0]
        if isinstance(expr_arg, ast.Constant) and isinstance(expr_arg.value, str):
            is_allowed, error_msg = _validate_math_eval_expression_source(expr_arg.value)
            if not is_allowed:
                return False, f"eval 数学表达式不安全：{error_msg}"
        return True, ""

    def _validate_math_expression_tree(self, tree: ast.AST) -> Tuple[bool, str]:
        return _validate_math_eval_expression_tree(tree)

    def _validate_math_expression_call(self, node: ast.Call) -> Tuple[bool, str]:
        return _validate_math_eval_expression_call(node)

    def _validate_import(self, module_name: str) -> Tuple[bool, str]:
        root = _root_module_name(module_name)
        if root == "sys":
            return True, ""
        if root in FORBIDDEN_MODULES:
            return False, f"禁止导入模块：{module_name}"
        if root not in ALLOWED_MODULES:
            return False, f"未授权的模块：{module_name}"
        return True, ""

    def _failure(
        self,
        status: str,
        error: str,
        *,
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "output": "",
            "error": error,
            "status": status,
            "stderr": "",
            "exit_code": exit_code,
            "duration": duration,
        }

    def _get_sandbox_input_text(self) -> Tuple[str, bool, str | None]:
        for key in SANDBOX_INPUT_CONTEXT_KEYS:
            if key in self.context:
                try:
                    data = self._to_jsonable(self.context[key])
                    return json.dumps(data, ensure_ascii=False), True, None
                except (TypeError, ValueError) as exc:
                    return "", False, f"无法序列化沙盒输入数据：{type(exc).__name__}: {exc}"
        return "", False, None

    def _extract_code(self, code: str) -> str:
        raw_code = str(code).strip()
        if "```python" in raw_code:
            return raw_code.split("```python")[-1].split("```")[0].strip()
        if "```" not in raw_code:
            return raw_code

        parts = raw_code.split("```")
        if len(parts) <= 1:
            return raw_code
        potential_code = parts[1]
        if "\n" not in potential_code:
            return potential_code.strip()
        first_line, rest_of_code = potential_code.split("\n", 1)
        if first_line.strip().isalpha():
            return rest_of_code.strip()
        return potential_code.strip()

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            return cls._to_jsonable(value.tolist())
        if hasattr(value, "item"):
            return cls._to_jsonable(value.item())
        raise TypeError(f"unsupported data type {type(value).__name__}")

    @staticmethod
    def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _terminate_process(process: mp.Process) -> None:
        process.terminate()
        process.join(1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join()
