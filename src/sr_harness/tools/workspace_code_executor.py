# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工作区代码执行工具。

继承 CodeExecutorTool，在工作区目录中执行 Python 代码，允许对工作区内文件
进行读写操作。通过路径约束机制确保所有文件操作限制在工作区范围内。
"""
from __future__ import annotations
import io
import os
import time
import json
import queue
import signal
import builtins
import traceback
import contextlib
import multiprocessing as mp
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from .base_tool import BaseTool, ToolMetadata
from .code_executor import (
    CodeExecutorTool,
    LimitedWriter,
    SandBoxCodeExecutor,
    sandbox_builtin,
    sandbox_module,
)


class WorkspaceSandBoxCodeExecutor(SandBoxCodeExecutor):
    """允许工作区内文件访问的沙箱执行器。"""

    ALLOWED_MODULES = SandBoxCodeExecutor.ALLOWED_MODULES | {"csv", "pandas", "io"}
    FORBIDDEN_CALLS = SandBoxCodeExecutor.FORBIDDEN_CALLS - {"open"}

    @classmethod
    def check_workspace_path(cls, path, workspace_dir: str) -> str:
        """校验路径是否在工作区内，返回规范化的绝对路径字符串。"""
        workspace = Path(workspace_dir).resolve()
        target = Path(str(path))
        if not target.is_absolute():
            target = workspace / target
        resolved = target.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError:
            raise PermissionError(f"禁止访问工作区外的路径：{path}")
        return str(resolved)

    @classmethod
    @sandbox_builtin("open")
    def make_restricted_open(cls, workspace_dir: str, **sandbox_context):
        """创建一个路径受限的 open() 函数。"""
        original_open = builtins.open

        def restricted_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
            checked = cls.check_workspace_path(file, workspace_dir)
            return original_open(checked, mode, buffering, encoding, errors, newline)

        def install():
            # Patch process-wide file entry points so library code (e.g. pandas/pathlib) also goes through path checks.
            builtins.open = restricted_open
            io.open = restricted_open

        restricted_open._sandbox_install = install
        return restricted_open

    @classmethod
    def make_blocked_func(cls, name: str):
        """创建一个调用时抛出 PermissionError 的占位函数。"""
        def blocked(*args, **kwargs):
            raise PermissionError(f"禁止调用 os.{name}()")
        blocked.__name__ = name
        return blocked

    @classmethod
    @sandbox_module("os")
    def make_restricted_os(cls, workspace_dir: str, **sandbox_context) -> ModuleType:
        """创建路径受限的 os 模块代理。"""
        proxy = ModuleType("os")
        proxy.__name__ = "os"
        proxy.__package__ = ""
        proxy.path = os.path
        proxy.sep = os.sep
        proxy.linesep = os.linesep
        proxy.curdir = os.curdir
        proxy.pardir = os.pardir
        proxy.devnull = os.devnull
        proxy.name = os.name
        proxy.cpu_count = os.cpu_count

        def check(path):
            return cls.check_workspace_path(path, workspace_dir)

        def getcwd():
            return os.getcwd()

        def listdir(path="."):
            check(path)
            return os.listdir(path)

        def scandir(path="."):
            check(path)
            return os.scandir(path)

        def stat(path, **kwargs):
            check(path)
            return os.stat(path, **kwargs)

        def mkdir(path, mode=0o777, **kwargs):
            check(path)
            return os.mkdir(path, mode, **kwargs)

        def makedirs(name, mode=0o777, exist_ok=False):
            check(name)
            return os.makedirs(name, mode, exist_ok=exist_ok)

        def remove(path):
            check(path)
            return os.remove(path)

        def unlink(path):
            check(path)
            return os.unlink(path)

        def rename(src, dst):
            check(src)
            check(dst)
            return os.rename(src, dst)

        def replace(src, dst):
            check(src)
            check(dst)
            return os.replace(src, dst)

        def rmdir(path):
            check(path)
            return os.rmdir(path)

        def walk(top, topdown=True, onerror=None, followlinks=False):
            check(top)
            return os.walk(top, topdown, onerror, followlinks)

        proxy.getcwd = getcwd
        proxy.listdir = listdir
        proxy.scandir = scandir
        proxy.stat = stat
        proxy.mkdir = mkdir
        proxy.makedirs = makedirs
        proxy.remove = remove
        proxy.unlink = unlink
        proxy.rename = rename
        proxy.replace = replace
        proxy.rmdir = rmdir
        proxy.walk = walk

        blocked_names = (
            "chdir", "fchdir", "chroot",
            "system", "popen", "popen2", "popen3", "popen4",
            "execl", "execle", "execlp", "execlpe",
            "execv", "execve", "execvp", "execvpe",
            "spawnl", "spawnle", "spawnlp", "spawnlpe",
            "spawnv", "spawnve", "spawnvp", "spawnvpe",
            "fork", "forkpty", "kill", "killpg",
            "link", "symlink",
        )
        for name in blocked_names:
            setattr(proxy, name, cls.make_blocked_func(name))

        return proxy

    @classmethod
    @sandbox_module("glob")
    def make_restricted_glob(cls, workspace_dir: str, **sandbox_context) -> ModuleType:
        """创建路径受限的 glob 模块代理。"""
        import glob as real_glob

        proxy = ModuleType("glob")
        proxy.__name__ = "glob"

        def glob(pathname, *, root_dir=None, dir_fd=None, recursive=False,
                 include_hidden=False):
            effective_root = root_dir or "."
            cls.check_workspace_path(effective_root, workspace_dir)
            if Path(pathname).is_absolute():
                cls.check_workspace_path(pathname, workspace_dir)
            return real_glob.glob(
                pathname, root_dir=root_dir, dir_fd=dir_fd,
                recursive=recursive, include_hidden=include_hidden,
            )

        def iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False,
                  include_hidden=False):
            effective_root = root_dir or "."
            cls.check_workspace_path(effective_root, workspace_dir)
            if Path(pathname).is_absolute():
                cls.check_workspace_path(pathname, workspace_dir)
            return real_glob.iglob(
                pathname, root_dir=root_dir, dir_fd=dir_fd,
                recursive=recursive, include_hidden=include_hidden,
            )

        proxy.glob = glob
        proxy.iglob = iglob
        proxy.escape = real_glob.escape
        return proxy

    @classmethod
    def sandbox_worker(
        cls,
        program: str,
        stdin_text: str,
        timeout_seconds: int,
        memory_limit_mb: int,
        output_limit_bytes: int,
        workspace_dir: str,
        result_queue: mp.Queue,
    ) -> None:
        # 准备受限环境
        cls.prepare_sandbox_runtime(
            stdin_text=stdin_text,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            workspace_dir=workspace_dir,
        )
        os.chdir(workspace_dir)

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


@BaseTool.register("workspace_code_executor")
class WorkspaceCodeExecutorTool(CodeExecutorTool):
    metadata = ToolMetadata(name="workspace_code_executor")

    def execute(
        self,
        program: str,
        timeout_seconds: int = CodeExecutorTool.DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = CodeExecutorTool.DEFAULT_MEMORY_LIMIT_MB,
        output_limit_bytes: int = CodeExecutorTool.DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> Dict[str, Any]:
        """Execute Python code in the workspace directory with file access.
        1) Code runs with cwd set to the workspace directory.
        2) Use `open("filename")` or `pandas.read_csv("filename")` to read workspace files.
        3) Use `open("output.csv", "w")` to write results back to the workspace.
        4) All file paths must be within the workspace. Absolute paths outside workspace are forbidden.
        5) numpy, scipy, pandas, csv are available. Network and subprocess modules are forbidden.
        6) exec() is forbidden. eval() is allowed only for math expression strings.

        Args:
            program: Python code string to execute.
            timeout_seconds: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.
            output_limit_bytes: Limit on the amount of output (in bytes) that can be produced.
        """
        # 准备 stdin (已被弃用)
        if not hasattr(self, 'stdin_text'):
            assert 'data' in self.context
            data = self.context['data']
            data_dict = self.serialization(data)
            stdin_text = self.stdin_text = json.dumps(data_dict, ensure_ascii=False)
        else:
            stdin_text = self.stdin_text

        # 准备 program
        program = self.extract_code(program)
        if not (validation_result := WorkspaceSandBoxCodeExecutor.validate_code(program))['is_safe']:
            raise Exception(f"Code security check failed since: {validation_result['error_msg']}")

        # 准备子进程
        assert 'workspace_dir' in self.context
        workspace_dir = self.context['workspace_dir']
        timeout_seconds = self.bounded_int(timeout_seconds, self.DEFAULT_TIMEOUT_SECONDS, 1, self.MAX_TIMEOUT_SECONDS)
        memory_limit_mb = self.bounded_int(memory_limit_mb, self.DEFAULT_MEMORY_LIMIT_MB, 64, self.MAX_MEMORY_LIMIT_MB)
        output_limit_bytes = self.bounded_int(output_limit_bytes, self.DEFAULT_OUTPUT_LIMIT_BYTES, 1024, self.MAX_OUTPUT_LIMIT_BYTES)
        mp_context = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")  # Windows (nt) 不支持 fork, 必须用 spawn
        result_queue = mp_context.Queue(maxsize=1)
        worker_args = (program, stdin_text, timeout_seconds, memory_limit_mb, output_limit_bytes, workspace_dir, result_queue)
        process = mp_context.Process(target=WorkspaceSandBoxCodeExecutor.sandbox_worker, args=worker_args)

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
