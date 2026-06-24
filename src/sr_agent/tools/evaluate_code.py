# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码模型评估工具。
用受限 Python 代码构建模型并生成预测，用于补充 nd2py 公式表达能力。
"""
from __future__ import annotations

import os
import ast
import time
import queue
import signal
import builtins
import traceback
import contextlib
import numpy as np
import nd2py as nd
import multiprocessing as mp
from typing import Any, Dict
from ..utils import log_exception
from .base_tool import BaseTool, ToolMetadata
from .code_executor import CodeExecutorTool, LimitedWriter, SandBoxCodeExecutor


@BaseTool.register("evaluate_code")
class EvaluateCodeTool(BaseTool):
    metadata = ToolMetadata(name="evaluate_code")

    DEFAULT_TIMEOUT_SECONDS = CodeExecutorTool.DEFAULT_TIMEOUT_SECONDS
    DEFAULT_MEMORY_LIMIT_MB = CodeExecutorTool.DEFAULT_MEMORY_LIMIT_MB
    DEFAULT_OUTPUT_LIMIT_BYTES = CodeExecutorTool.DEFAULT_OUTPUT_LIMIT_BYTES
    MAX_TIMEOUT_SECONDS = CodeExecutorTool.MAX_TIMEOUT_SECONDS
    MAX_MEMORY_LIMIT_MB = CodeExecutorTool.MAX_MEMORY_LIMIT_MB
    MAX_OUTPUT_LIMIT_BYTES = CodeExecutorTool.MAX_OUTPUT_LIMIT_BYTES

    def execute(
        self,
        model_code: str,
        predict_code: str,
        format_code: str = None,
        y: str = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> Dict[str, Any]:
        """Evaluate a Python-defined candidate model on the current dataset.

        Use this tool when a candidate cannot be expressed conveniently as an nd2py formula.
        The code runs in a restricted sandbox, then the tool computes metrics against the target
        and returns the formatted model under the `formula` key.

        Args:
            model_code: Code containing exactly one function with signature `def func(data)` plus optional top-level imports. 
                `data` is a dictionary mapping variable names to numeric arrays, including the target variable.
                This function should return a model object that will be passed to `predict_code` and `format_code`.
            predict_code: Code containing exactly one function with signature `def func(data, model)` plus optional top-level imports.
                The function should return the predicted value for the target `y`, 
                which must be array-like and compatible with the target shape.
            format_code: Code containing exactly one function with signature `def func(model)` plus optional top-level imports. 
                The function should return a concise readable string representation of the model.
                This argument is optional; omit it unless a custom display string is needed.
                If omitted, uses `str(model)` or `repr(model)` as a fallback.
            y: Target variable name. Use target variable by default.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            timeout_seconds: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.
            output_limit_bytes: Limit on the amount of output (in bytes) that can be produced.
        """
        data = self.context['data']
        y = y or self.context['target']
        y = y.strip().strip('"').strip("'")
        eq_y = nd.parse(
            y.replace("^", "**").replace('np.', '').replace('math.', ''),
            variables={'pi': np.pi, 'e': np.e},
        )
        y_true = eq_y.eval(data)

        format_code = format_code or (
            f"def format_code(model):\n"
            f"    try:\n"
            f"        return str(model)\n"
            f"    except Exception:\n"
            f"        return repr(model)\n"
        )
        timeout_seconds = CodeExecutorTool.bounded_int(timeout_seconds, self.DEFAULT_TIMEOUT_SECONDS, 1, self.MAX_TIMEOUT_SECONDS)
        memory_limit_mb = CodeExecutorTool.bounded_int(memory_limit_mb, self.DEFAULT_MEMORY_LIMIT_MB, 64, self.MAX_MEMORY_LIMIT_MB)
        output_limit_bytes = CodeExecutorTool.bounded_int(output_limit_bytes, self.DEFAULT_OUTPUT_LIMIT_BYTES, 1024, self.MAX_OUTPUT_LIMIT_BYTES)

        prepared_model_code, model_func_name = self.prepare_function_code(model_code, ("data",), "model_code")
        prepared_predict_code, predict_func_name = self.prepare_function_code(predict_code, ("data", "model"), "predict_code")
        prepared_format_code, format_func_name = self.prepare_function_code(format_code, ("model",), "format_code")

        mp_context = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
        result_queue = mp_context.Queue(maxsize=1)
        process = mp_context.Process(target=self.sandbox_worker, args=(
            prepared_model_code, model_func_name,
            prepared_predict_code, predict_func_name,
            prepared_format_code, format_func_name,
            self.context['data'], self.context['target'],
            timeout_seconds, memory_limit_mb, output_limit_bytes,
            result_queue,
        ))

        max_retry = 3
        for attempt in range(1, max_retry + 1):
            try:
                process.start()
                break
            except RuntimeError as e:
                retryable = "can't start new thread" in str(e) or "Resource temporarily unavailable" in str(e)
                if not retryable:
                    raise Exception(f"Cannot start sandbox subprocess: {log_exception(e, with_traceback=False)}") from e
                if attempt < max_retry:
                    time.sleep(1.0 * attempt)
        else:
            raise Exception(f"Cannot start sandbox subprocess after {max_retry} attempts: {log_exception(e, with_traceback=False)}")

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

        if result is not None:
            if result['sandbox_error'] is not None:
                raise Exception(f"Sandbox execution error: {result['sandbox_error']}")
            process.join(1)
            CodeExecutorTool.terminate_process(process)
        elif process.is_alive():
            CodeExecutorTool.terminate_process(process)
            raise Exception(f"Sandbox subprocess did not return result before timeout={timeout_seconds} seconds and has been terminated.")
        elif (exit_code := process.exitcode) is not None and exit_code < 0:
            raise Exception(f"Sandbox subprocess was terminated by signal {signal.Signals(-exit_code).name}.")
        else:
            raise Exception(f"Sandbox subprocess did not return result and has exited with code {process.exitcode}.")

        y_pred = np.asarray(result["y_pred"])
        return {
            "formula": result["model_str"],
            "metrics": self.evaluate(y_pred=y_pred, y_true=y_true),
            "is_candidate": result["is_candidate"] and (y == self.context['target']),
        }

    @classmethod
    def prepare_function_code(cls, code: str, expected_params: tuple[str, ...], code_name: str) -> tuple[str, str]:
        """ 将代码字符串解析为单个函数定义，并验证其签名和安全性。 """
        if not (code := CodeExecutorTool.extract_code(code)):
            raise ValueError(f"{code_name} must not be empty.")
        if not (validation_result := SandBoxCodeExecutor.validate_code(code))["is_safe"]:
            raise Exception(f"Code security check failed since: {validation_result['error_msg']}")
        tree = ast.parse(code)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError(f"{code_name} must contain exactly one function definition.")
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                raise ValueError(f"{code_name} may only contain top-level imports and one function definition.")
        function = functions[0]
        if function.decorator_list:
            raise ValueError(f"{code_name} function decorators are not allowed.")
        args = function.args
        if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
            raise ValueError(f"{code_name} function must use only ordinary positional arguments.")
        if args.defaults:
            raise ValueError(f"{code_name} function arguments must not have defaults.")
        param_names = tuple(arg.arg for arg in args.args)
        if param_names != expected_params:
            expected = ", ".join(expected_params)
            actual = ", ".join(param_names)
            raise ValueError(f"{code_name} function signature must be ({expected}), got ({actual}).")
        return code, function.name

    @classmethod
    def sandbox_worker(
        cls,
        model_code: str,
        model_func_name: str,
        predict_code: str,
        predict_func_name: str,
        format_code: str,
        format_func_name: str,
        data: Dict[str, Any],
        target: str,
        timeout_seconds: int,
        memory_limit_mb: int,
        output_limit_bytes: int,
        result_queue: mp.Queue,
    ) -> None:
        SandBoxCodeExecutor.prepare_sandbox_runtime(
            stdin_text="",
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
        stdout = LimitedWriter(output_limit_bytes)
        stderr = LimitedWriter(output_limit_bytes)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                # Prepare safe_globals
                safe_builtins = {
                    name: getattr(builtins, name)
                    for name in SandBoxCodeExecutor.SAFE_BUILTIN_NAMES
                }
                safe_builtins["__import__"] = SandBoxCodeExecutor.limited_import
                safe_builtins.update(SandBoxCodeExecutor._SANDBOX_BUILTINS)
                safe_globals = {
                    "__builtins__": safe_builtins,
                    "__name__": "__sandbox__",
                    "__package__": None,
                    "data": data,
                    "np": np,
                    "numpy": np,
                }

                model = cls.call_code_function(model_code, model_func_name, (data,), "<evaluate-code-model>", safe_globals)
                y_pred = cls.call_code_function(predict_code, predict_func_name, (data, model), "<evaluate-code-predict>", safe_globals)
                model_str = cls.call_code_function(format_code, format_func_name, (model,), "<evaluate-code-format>", safe_globals)

                try:
                    candidate_data = dict(data)
                    candidate_data.pop(target, None)
                    tmp = cls.call_code_function(predict_code, predict_func_name, (candidate_data, model), "<evaluate-code-candidate>", safe_globals)
                    tmp = np.asarray(tmp) # 检查输出是否可以转换为数组
                    is_candidate = True
                except Exception:
                    is_candidate = False

            result_queue.put({
                "y_pred": np.asarray(y_pred),
                "model_str": model_str,
                "is_candidate": is_candidate,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "sandbox_error": None,
            })
        except BaseException as e:
            stderr.write(traceback.format_exc(limit=8))
            result_queue.put({
                "y_pred": None,
                "model_str": "",
                "is_candidate": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "sandbox_error": f"{type(e).__name__}: {e}",
            })

    @classmethod
    def call_code_function(cls, code: str, function_name: str, inputs: tuple[Any, ...], filename: str, safe_globals) -> Any:
        before = safe_globals.get(function_name)
        exec(compile(code, filename, "exec"), safe_globals, safe_globals)
        function = safe_globals.get(function_name)
        if function is before or not callable(function):
            raise ValueError(f"{filename} must define callable function `{function_name}`.")
        return function(*inputs)
