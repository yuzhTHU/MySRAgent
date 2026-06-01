# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工作区 Shell 工具。

提供一个隔离的工作区目录，允许 Agent 通过受限的 shell 风格命令操作文件，
并通过 `python script.py` 执行工作区内的 Python 脚本。
"""
from __future__ import annotations

import os
import gzip
import shlex
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from .base_tool import BaseTool, ToolMetadata
from .code_executor import _LimitedWriter
from ..utils import log_exception

_logger = logging.getLogger(f"sr_agent.{__name__}")

# WorkspaceShellTool 支持的命令白名单
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "wc",
    "grep", "sort", "cut",
    "cp", "mv", "rm", "mkdir",
    "gunzip", "gzip", "unzip", "tar",
}


class Workspace:
    """管理一个隔离的临时工作区目录。

    初始化时将指定文件/目录以只读方式链接（或复制）到工作区内。
    提供路径解析和安全校验，严格防止路径逃逸。
    """

    def __init__(self, workspace_files: List[str] | None = None, temp_dir: str | None = None):
        self._path = Path(tempfile.mkdtemp(prefix="sr_workspace_", dir=temp_dir))
        _logger.info(f"Created workspace at {self._path}")
        for src in (workspace_files or []):
            self.link_item(Path(src))

    @property
    def path(self) -> Path:
        return self._path

    def resolve(self, relative_path: str) -> Path | None:
        """将相对路径解析为工作区内的绝对路径，返回 None 表示路径不合法。
        - None -> 工作区根目录
        - 合法相对路径 -> 工作区内的绝对路径
        - 非法路径（绝对路径、路径逃逸、符号链接指向工作区外）-> None
        """
        if not relative_path:
            return self._path
        # 拒绝 POSIX 绝对路径
        if os.path.isabs(relative_path):
            return None
        # 拒绝 Windows 驱动器号 (C:)
        if len(relative_path) >= 2 and relative_path[0].isalpha() and relative_path[1] == ":":
            return None
        candidate = (self._path / relative_path).resolve()
        # 拒绝路径逃逸（解析后的路径不在工作区内）
        try:
            candidate.relative_to(self._path.resolve())
        except ValueError:
            return None
        return candidate

    def cleanup(self):
        if self._path.exists():
            shutil.rmtree(self._path, ignore_errors=True)
            _logger.info(f"Cleaned up workspace at {self._path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def link_item(self, src: Path):
        """将文件或目录链接/复制到工作区内，设置为只读。"""
        src = src.resolve()
        dst = self._path / src.name
        if dst.exists():
            _logger.warning(f"Workspace item {dst.name} already exists, skipping.")
            return
        try:
            if src.is_dir():
                # 对目录创建符号链接（只读由目录内文件权限保证）
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError as e:
                    file_num = sum(1 for f in src.rglob("*") if f.is_file())
                    dir_size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
                    _logger.warning(
                        f"Failed to symlink {src} to workspace: {e}"
                        f" (is_dir={src.is_dir()}, file_num={file_num}, total_size={dir_size:,} bytes), falling back to copy."
                    )
                    shutil.copytree(src, dst)
            else:
                # 对文件优先硬链接，失败时复制
                try:
                    os.link(src, dst)
                except (OSError, NotImplementedError):
                    file_size = src.stat().st_size
                    _logger.warning(f"Failed to link {src} to workspace (size={file_size:,} bytes), falling back to copy.")
                    shutil.copy2(src, dst)
                # 设置只读
                os.chmod(dst, 0o444)
        except Exception as e:
            _logger.error(f"Failed to add {src} to workspace: {log_exception(e, with_traceback=False)}")


@BaseTool.register("workspace_shell")
class WorkspaceShellTool(BaseTool):
    metadata = ToolMetadata(name="workspace_shell")

    DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024
    MAX_OUTPUT_LIMIT_BYTES = 1024 * 1024

    def execute(
        self,
        command: str,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> Dict[str, Any]:
        """Execute a restricted shell command in the workspace directory.
        Supported commands: ls, cat, head, tail, wc, grep, sort, cut, cp, mv, rm, mkdir,
        gunzip, gzip, unzip, tar.
        All file paths are relative to the workspace root. Absolute paths and path traversal
        (e.g., ../) are forbidden.

        Args:
            command: A shell command string. 
                Examples: "ls", "cat data.csv | head -5", "gunzip data.csv.gz".
            output_limit_bytes: Maximum stdout size returned by each command segment.
        """
        workspace: Workspace = self.context.get("workspace")
        if workspace is None:
            return self._error("Workspace not initialized.")
        output_limit_bytes = self._bounded_output_limit(output_limit_bytes)

        # 处理管道：分割为多个命令，顺序执行，前一个的 stdout 作为后一个的 stdin
        pipe_segments = [seg.strip() for seg in command.split("|")]
        stdin_text = ""
        for segment in pipe_segments:
            try:
                result = self.execute_single(segment, workspace, stdin_text)
                if not result.get("success", False):
                    return result
                stdin_text = self._limit_output(result.get("stdout", ""), output_limit_bytes)
            except Exception as e:
                _logger.error(f"Error executing command segment '{segment}': {log_exception(e)}")
                return self._error(f"Error executing command segment '{segment}', ask human for help: {e}")
        return self._ok(stdin_text)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"])
        if result.get("stderr"):
            parts.append(f"[stderr] {result['stderr']}")
        if result.get("error"):
            parts.append(f"[error] {result['error']}")
        return "\n".join(parts) if parts else "(no output)"

    def execute_single(self, command: str, workspace: Workspace, stdin_text: str) -> Dict[str, Any]:
        """执行单个命令（管道拆分后的一段）。"""
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return self._error(f"Command parse error: {e}")
        if not parts:
            return self._error("Empty command.")

        cmd_name = parts[0]
        args = parts[1:]

        if cmd_name not in ALLOWED_COMMANDS:
            return self._error(
                f"Command '{cmd_name}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        # 路由到对应的处理函数
        if (handler := getattr(self, f"_cmd_{cmd_name}")) is None:
            return self._error(f"Command '{cmd_name}' is not implemented.")
        else:
            return handler(args, workspace, stdin_text)

    @staticmethod
    def _ok(stdout: str) -> Dict[str, Any]:
        return {"success": True, "stdout": stdout, "stderr": "", "exit_code": 0}

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {"success": False, "stdout": "", "stderr": "", "error": msg, "exit_code": 1}

    @classmethod
    def _bounded_output_limit(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = cls.DEFAULT_OUTPUT_LIMIT_BYTES
        return max(1024, min(cls.MAX_OUTPUT_LIMIT_BYTES, parsed))

    @staticmethod
    def _limit_output(stdout: str, output_limit_bytes: int) -> str:
        writer = _LimitedWriter(output_limit_bytes)
        writer.write(stdout)
        return writer.getvalue()

    def _cmd_ls(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        targets = args if args else ["."]
        sections = []
        for target in targets:
            if (path := ws.resolve(target)) is None:
                return self._error(f"Invalid path: {target}")
            if not path.exists():
                return self._error(f"No such file or directory: {target}")
            if path.is_dir():
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
                if len(targets) > 1:
                    sections.append(f"{target}:\n" + "\n".join(entries))
                else:
                    sections.append("\n".join(entries))
            else:
                sections.append(path.name)
        return self._ok("\n\n".join(sections))

    def _cmd_cat(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._ok(stdin)
        parts: List[str] = []
        for arg in args:
            if (path := ws.resolve(arg)) is None:
                return self._error(f"Invalid path: {arg}")
            if not path.exists():
                return self._error(f"No such file: {arg}")
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                return self._error(f"Cannot read file: {e}")
        return self._ok("".join(parts))

    def _cmd_head(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        n = 10
        file_args = []
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args) and args[i + 1].isdigit():
                n = int(args[i + 1])
                i += 2
            elif args[i].startswith("-") and args[i][1:].isdigit():
                n = int(args[i][1:])
                i += 1
            else:
                file_args.append(args[i])
                i += 1
        if file_args:
            path = ws.resolve(file_args[0])
            if path is None:
                return self._error(f"Invalid path: {file_args[0]}")
            if not path.exists():
                return self._error(f"No such file: {file_args[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()[:n]
        return self._ok("\n".join(lines))

    def _cmd_tail(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        n = 10
        file_args = []
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args) and args[i + 1].isdigit():
                n = int(args[i + 1])
                i += 2
            elif args[i].startswith("-") and args[i][1:].isdigit():
                n = int(args[i][1:])
                i += 1
            else:
                file_args.append(args[i])
                i += 1
        if file_args:
            path = ws.resolve(file_args[0])
            if path is None:
                return self._error(f"Invalid path: {file_args[0]}")
            if not path.exists():
                return self._error(f"No such file: {file_args[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()[-n:]
        return self._ok("\n".join(lines))

    def _cmd_wc(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if args:
            path = ws.resolve(args[0])
            if path is None:
                return self._error(f"Invalid path: {args[0]}")
            if not path.exists():
                return self._error(f"No such file: {args[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()
        words = text.split()
        chars = len(text)
        return self._ok(f"{len(lines)} {len(words)} {chars}")

    def _cmd_grep(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._error("grep requires a pattern argument.")
        pattern = args[0]
        remaining = args[1:]
        if remaining:
            path = ws.resolve(remaining[0])
            if path is None:
                return self._error(f"Invalid path: {remaining[0]}")
            if not path.exists():
                return self._error(f"No such file: {remaining[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        matched = [line for line in text.splitlines() if pattern in line]
        return self._ok("\n".join(matched))

    def _cmd_sort(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if args:
            path = ws.resolve(args[0])
            if path is None:
                return self._error(f"Invalid path: {args[0]}")
            if not path.exists():
                return self._error(f"No such file: {args[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = sorted(text.splitlines())
        return self._ok("\n".join(lines))

    def _cmd_cut(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        # 简单实现：-d 分隔符 -f 字段
        delimiter = "\t"
        fields = None
        remaining = []
        i = 0
        while i < len(args):
            if args[i] == "-d" and i + 1 < len(args):
                delimiter = args[i + 1]
                i += 2
            elif args[i] == "-f" and i + 1 < len(args):
                fields = args[i + 1]
                i += 2
            else:
                remaining.append(args[i])
                i += 1
        if remaining:
            path = ws.resolve(remaining[0])
            if path is None:
                return self._error(f"Invalid path: {remaining[0]}")
            if not path.exists():
                return self._error(f"No such file: {remaining[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        if fields is None:
            return self._ok(text)
        # 解析字段索引（1-based）
        try:
            field_indices = [int(f) - 1 for f in fields.split(",")]
        except ValueError:
            return self._error(f"Invalid field specification: {fields}")
        output_lines = []
        for line in text.splitlines():
            parts = line.split(delimiter)
            selected = [parts[i] if i < len(parts) else "" for i in field_indices]
            output_lines.append(delimiter.join(selected))
        return self._ok("\n".join(output_lines))

    def _cmd_cp(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if len(args) < 2:
            return self._error("cp requires source and destination arguments.")
        src_path = ws.resolve(args[0])
        dst_path = ws.resolve(args[1])
        if src_path is None:
            return self._error(f"Invalid source path: {args[0]}")
        if dst_path is None:
            return self._error(f"Invalid destination path: {args[1]}")
        if not src_path.exists():
            return self._error(f"No such file: {args[0]}")
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            return self._ok("")
        except Exception as e:
            return self._error(f"cp failed: {e}")

    def _cmd_mv(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if len(args) < 2:
            return self._error("mv requires source and destination arguments.")
        src_path = ws.resolve(args[0])
        dst_path = ws.resolve(args[1])
        if src_path is None:
            return self._error(f"Invalid source path: {args[0]}")
        if dst_path is None:
            return self._error(f"Invalid destination path: {args[1]}")
        if not src_path.exists():
            return self._error(f"No such file: {args[0]}")
        try:
            shutil.move(str(src_path), str(dst_path))
            return self._ok("")
        except Exception as e:
            return self._error(f"mv failed: {e}")

    def _cmd_rm(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._error("rm requires a file argument.")
        path = ws.resolve(args[0])
        if path is None:
            return self._error(f"Invalid path: {args[0]}")
        if not path.exists():
            return self._error(f"No such file: {args[0]}")
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return self._ok("")
        except Exception as e:
            return self._error(f"rm failed: {e}")

    def _cmd_mkdir(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._error("mkdir requires a directory name.")
        path = ws.resolve(args[0])
        if path is None:
            return self._error(f"Invalid path: {args[0]}")
        try:
            path.mkdir(parents=True, exist_ok=True)
            return self._ok("")
        except Exception as e:
            return self._error(f"mkdir failed: {e}")

    def _cmd_gunzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._error("gunzip requires a file argument.")
        path = ws.resolve(args[0])
        if path is None:
            return self._error(f"Invalid path: {args[0]}")
        if not path.exists():
            return self._error(f"No such file: {args[0]}")
        out_path = path.with_suffix("") if path.suffix == ".gz" else path.parent / (path.name + ".out")
        try:
            with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            path.unlink()
            return self._ok(f"Decompressed to {out_path.name}")
        except Exception as e:
            return self._error(f"gunzip failed: {e}")

    def _cmd_gzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        if not args:
            return self._error("gzip requires a file argument.")
        path = ws.resolve(args[0])
        if path is None:
            return self._error(f"Invalid path: {args[0]}")
        if not path.exists():
            return self._error(f"No such file: {args[0]}")
        out_path = path.parent / (path.name + ".gz")
        try:
            with open(path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            path.unlink()
            return self._ok(f"Compressed to {out_path.name}")
        except Exception as e:
            return self._error(f"gzip failed: {e}")

    def _cmd_unzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        import zipfile
        if not args:
            return self._error("unzip requires a file argument.")
        path = ws.resolve(args[0])
        if path is None:
            return self._error(f"Invalid path: {args[0]}")
        if not path.exists():
            return self._error(f"No such file: {args[0]}")
        try:
            with zipfile.ZipFile(path, "r") as zf:
                # 安全检查：确保 zip 内的路径不会逃逸
                for name in zf.namelist():
                    if name.startswith("/") or ".." in name:
                        return self._error(f"Unsafe path in archive: {name}")
                zf.extractall(ws.path)
            return self._ok(f"Extracted to workspace root.")
        except Exception as e:
            return self._error(f"unzip failed: {e}")

    def _cmd_tar(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        import tarfile
        if not args:
            return self._error("tar requires arguments (e.g., tar -xf file.tar.gz).")
        # 简单解析：支持 -xf / -xzf
        flags = ""
        file_arg = None
        for a in args:
            if a.startswith("-"):
                flags += a.lstrip("-")
            elif file_arg is None:
                file_arg = a
        if file_arg is None:
            return self._error("tar requires a file argument.")
        path = ws.resolve(file_arg)
        if path is None:
            return self._error(f"Invalid path: {file_arg}")
        if not path.exists():
            return self._error(f"No such file: {file_arg}")
        if "x" not in flags:
            return self._error("Only tar extraction (-x) is supported.")
        mode = "r:gz" if "z" in flags else "r:*"
        try:
            with tarfile.open(path, mode) as tf:
                # 安全检查
                for member in tf.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        return self._error(f"Unsafe path in archive: {member.name}")
                tf.extractall(ws.path)
            return self._ok(f"Extracted to workspace root.")
        except Exception as e:
            return self._error(f"tar failed: {e}")
