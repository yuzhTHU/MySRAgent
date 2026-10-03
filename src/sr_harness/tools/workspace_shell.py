# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工作区 Shell 工具。

提供一个隔离的工作区目录，允许 Agent 通过受限的 shell 风格命令操作
文件。
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import shlex
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Never

from .base_tool import BaseTool, ToolMetadata
from .code_executor import LimitedWriter
from ..utils import log_exception

_logger = logging.getLogger(f"sr_harness.{__name__}")

# WorkspaceShellTool 支持的命令白名单
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "wc",
    "grep", "sort", "cut",
    "cp", "mv", "rm", "mkdir",
    "gunzip", "gzip", "unzip", "tar",
}


class CommandParseError(ValueError):
    """Raised when the restricted command language cannot parse an invocation."""


class CommandArgumentParser(argparse.ArgumentParser):
    """An ``argparse`` parser that reports errors instead of exiting the process."""

    def __init__(self, command: str):
        super().__init__(prog=command, add_help=False, allow_abbrev=False, exit_on_error=False)

    def error(self, message: str) -> Never:
        raise CommandParseError(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        raise CommandParseError(message or f"{self.prog}: invalid arguments")


class Workspace:
    """管理一个隔离的临时工作区目录。

    初始化时将指定文件/目录以只读方式链接（或复制）到工作区内。
    提供路径解析和安全校验，严格防止路径逃逸。
    """

    def __init__(self, workspace_files: List[str] | None = None, temp_dir: str | None = None):
        self.retain = False
        self._path = Path(tempfile.mkdtemp(prefix="sr_workspace_", dir=temp_dir))
        _logger.info(f"Created workspace at {self._path}")
        for src in (workspace_files or []):
            self.link_item(Path(src))

    @property
    def path(self) -> Path:
        return self._path

    def resolve(self, relative_path: str) -> Path | None:
        """将相对路径解析为工作区内的绝对路径。

        返回 None 表示路径不合法。
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
        if not self.retain:
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
                        f" (is_dir={src.is_dir()}, file_num={file_num}, "
                        f"total_size={dir_size:,} bytes), falling back to copy."
                    )
                    shutil.copytree(src, dst)
            else:
                # 对文件优先硬链接，失败时复制
                try:
                    os.link(src, dst)
                except (OSError, NotImplementedError):
                    file_size = src.stat().st_size
                    _logger.warning(
                        f"Failed to link {src} to workspace "
                        f"(size={file_size:,} bytes), falling back to copy."
                    )
                    shutil.copy2(src, dst)
                # 设置只读
                os.chmod(dst, 0o444)
        except Exception as e:
            _logger.error(
                f"Failed to add {src} to workspace: "
                f"{log_exception(e, with_traceback=False)}"
            )


@BaseTool.register("workspace_shell")
class WorkspaceShellTool(BaseTool):
    metadata = ToolMetadata(name="workspace_shell")

    DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024
    MAX_OUTPUT_LIMIT_BYTES = 1024 * 1024

    @classmethod
    def get_doc(cls) -> dict[str, str]:
        return {
            "name": "workspace-shell",
            "description": (
                "Inspect and modify workspace files with a safe, restricted shell-like "
                "command language."
            ),
            "content": (
                "# Workspace Shell\n\n"
                "`workspace_shell` implements a small command language in Python. It does not "
                "start a system shell or execute operating-system commands. Use it to inspect, "
                "filter, copy, move, remove, and unpack files in the current workspace.\n\n"
                "## Syntax\n\n"
                "- Separate arguments with whitespace. Single quotes, double quotes, and "
                "backslash escapes follow POSIX shell lexical rules.\n"
                "- Join supported commands with `|`. A pipeline passes each command's text "
                "output to the next command in memory; spaces around `|` are optional.\n"
                "- Short flags may be combined or separated: `ls -la`, `ls -al`, "
                "`ls -l -a`, and `ls -a -l` are equivalent. Options and operands may usually "
                "be interspersed. Use `--` before a path beginning with `-`.\n"
                "- Every path must resolve inside the workspace. Absolute paths and paths or "
                "symbolic links that escape the workspace are rejected.\n"
                "- Redirection and other shell operators (`>`, `<`, `;`, `&&`, `||`, `&`) are "
                "not supported. There is no globbing, variable or tilde expansion, command "
                "substitution, background execution, or invocation of unlisted commands.\n\n"
                "## Supported commands\n\n"
                "```text\n"
                "ls [-a|--all] [-l] [PATH ...]\n"
                "cat [-n|--number] [FILE ...]\n"
                "head [-n N|--lines N|-N] [FILE]\n"
                "tail [-n N|--lines N|-N] [FILE]\n"
                "wc [-l] [-w] [-c] [-m] [FILE ...]\n"
                "grep [-i|--ignore-case] [-v|--invert-match] [-n|--line-number]\n"
                "     [-c|--count] [-F|--fixed-strings] PATTERN [FILE]\n"
                "sort [-r|--reverse] [-n|--numeric-sort] [-u|--unique] [FILE]\n"
                "cut [-d DELIMITER|--delimiter DELIMITER] -f FIELDS [FILE]\n"
                "cp [-r|-R|--recursive] [-f|--force] SOURCE DESTINATION\n"
                "mv [-f|--force] SOURCE DESTINATION\n"
                "rm [-r|-R|--recursive] [-f|--force] PATH ...\n"
                "mkdir [-p|--parents] DIRECTORY ...\n"
                "gzip [-k|--keep] [-f|--force] FILE\n"
                "gunzip [-k|--keep] [-f|--force] FILE\n"
                "unzip [-d DIRECTORY|--directory DIRECTORY] FILE\n"
                "tar (-x|--extract) [-z|--gzip] (-f FILE|--file FILE)\n"
                "    [-C DIRECTORY|--directory DIRECTORY]\n"
                "```\n\n"
                "`grep` performs fixed-string matching; `-F` is accepted for familiar grep "
                "syntax. `cut -f` accepts comma-separated, one-based field numbers. `tar` only "
                "supports extraction and accepts grouped flags such as `-xf` and `-xzf`. When "
                "a text-processing command has no file operand, it reads the previous pipeline "
                "stage.\n\n"
                "## Examples\n\n"
                "```text\n"
                "ls -la\n"
                "cat 'experiment 1.csv' | head -5\n"
                "cat data.csv|grep -i result|wc -l\n"
                "cut data.csv -d, -f1,3 | sort -u\n"
                "mkdir -p results/archive\n"
                "cp -r results results-copy\n"
                "tar -xzf results.tar.gz -C results/archive\n"
                "cat -- -notes\n"
                "```"
            ),
        }

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

        try:
            pipeline = self._parse_pipeline(command)
        except CommandParseError as exc:
            return self._error(f"Command parse error: {exc}")

        stdin_text = ""
        for tokens in pipeline:
            try:
                result = self._execute_tokens(tokens, workspace, stdin_text)
                if not result.get("success", False):
                    return result
                stdin_text = self._limit_output(result.get("stdout", ""), output_limit_bytes)
            except Exception as e:
                rendered = shlex.join(tokens)
                _logger.error(f"Error executing command segment '{rendered}': {log_exception(e)}")
                return self._error(
                    f"Error executing command segment '{rendered}', ask human for help: {e}"
                )
        return self._ok(stdin_text)

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if not result.get("success", False):
            detail = result.get("error") or result.get("stderr") or "No error detail was provided."
            return f"Command failed: {detail}"
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"])
        if result.get("stderr"):
            parts.append(f"Command stderr:\n{result['stderr']}")
        return (
            "\n".join(parts)
            if parts
            else "Command completed successfully and produced no output."
        )

    def execute_single(self, command: str, workspace: Workspace, stdin_text: str) -> Dict[str, Any]:
        """执行单个命令（管道拆分后的一段）。"""
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return self._error(f"Command parse error: {e}")
        return self._execute_tokens(parts, workspace, stdin_text)

    @staticmethod
    def _parse_pipeline(command: str) -> list[list[str]]:
        """Parse the deliberately small shell-like language into a pipeline AST.

        Quotes and backslash escapes follow POSIX shell lexical rules, but the only
        executable operator is a single pipe.  Shell evaluation, expansion,
        redirection, conditionals, command substitution, and background jobs are
        intentionally outside this language.
        """
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError as exc:
            raise CommandParseError(str(exc)) from exc

        pipeline: list[list[str]] = [[]]
        for token in tokens:
            if token == "|":
                if not pipeline[-1]:
                    raise CommandParseError("empty command before pipe")
                pipeline.append([])
            elif token and all(char in "|&;<>" for char in token):
                raise CommandParseError(f"operator {token!r} is not supported")
            else:
                pipeline[-1].append(token)
        if not pipeline[-1]:
            message = "empty command" if len(pipeline) == 1 else "empty command after pipe"
            raise CommandParseError(message)
        return pipeline

    def _execute_tokens(
        self, parts: list[str], workspace: Workspace, stdin_text: str
    ) -> Dict[str, Any]:
        if not parts:
            return self._error("Empty command.")

        cmd_name = parts[0]
        args = parts[1:]

        if cmd_name not in ALLOWED_COMMANDS:
            return self._error(
                f"Command '{cmd_name}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        handler = getattr(self, f"_cmd_{cmd_name}", None)
        if handler is None:
            return self._error(f"Command '{cmd_name}' is not implemented.")
        try:
            return handler(args, workspace, stdin_text)
        except CommandParseError as exc:
            return self._error(str(exc))

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
        writer = LimitedWriter(output_limit_bytes)
        writer.write(stdout)
        return writer.getvalue()

    @staticmethod
    def _parse_args(
        command: str,
        args: list[str],
        configure: Callable[[CommandArgumentParser], None],
    ) -> argparse.Namespace:
        parser = CommandArgumentParser(command)
        configure(parser)
        try:
            return parser.parse_intermixed_args(args)
        except argparse.ArgumentError as exc:
            raise CommandParseError(f"{command}: {exc}") from exc

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.is_file():
            raise CommandParseError(f"Not a regular file: {path.name}")
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CommandParseError(f"Cannot read file: {exc}") from exc

    @staticmethod
    def _nonnegative_int(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be non-negative")
        return parsed

    def _cmd_ls(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-a", "--all", action="store_true")
            parser.add_argument("-l", action="store_true", dest="long")
            parser.add_argument("paths", nargs="*")

        options = self._parse_args("ls", args, configure)
        targets = options.paths or ["."]
        sections = []
        for target in targets:
            if (path := ws.resolve(target)) is None:
                return self._error(f"Invalid path: {target}")
            if not path.exists():
                return self._error(f"No such file or directory: {target}")
            if path.is_dir():
                entries = sorted(
                    (
                        entry
                        for entry in path.iterdir()
                        if options.all or not entry.name.startswith(".")
                    ),
                    key=lambda entry: entry.name,
                )
                names = [
                    entry.name + ("/" if entry.is_dir() else "") for entry in entries
                ]
                if options.long:
                    rendered = []
                    for name in names:
                        entry = path / name.rstrip("/")
                        info = entry.stat()
                        modified = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
                        rendered.append(
                            f"{stat.filemode(info.st_mode)} {info.st_size:>10} {modified} {name}"
                        )
                    output = "\n".join(rendered)
                else:
                    output = "\n".join(names)
                if len(targets) > 1:
                    sections.append(f"{target}:\n" + output)
                else:
                    sections.append(output)
            else:
                if options.long:
                    info = path.stat()
                    modified = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
                    sections.append(
                        f"{stat.filemode(info.st_mode)} {info.st_size:>10} {modified} {path.name}"
                    )
                else:
                    sections.append(path.name)
        return self._ok("\n\n".join(sections))

    def _cmd_cat(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-n", "--number", action="store_true")
            parser.add_argument("files", nargs="*")

        options = self._parse_args("cat", args, configure)
        if not options.files:
            text = stdin
            if options.number:
                text = "\n".join(
                    f"{index:6}\t{line}" for index, line in enumerate(text.splitlines(), 1)
                )
            return self._ok(text)
        parts: List[str] = []
        for arg in options.files:
            if (path := ws.resolve(arg)) is None:
                return self._error(f"Invalid path: {arg}")
            if not path.exists():
                return self._error(f"No such file: {arg}")
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                return self._error(f"Cannot read file: {e}")
        text = "".join(parts)
        if options.number:
            text = "\n".join(
                f"{index:6}\t{line}" for index, line in enumerate(text.splitlines(), 1)
            )
        return self._ok(text)

    def _cmd_head(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        normalized = [
            part
            for arg in args
            for part in (["-n", arg[1:]] if re.fullmatch(r"-\d+", arg) else [arg])
        ]

        def configure(parser):
            parser.add_argument("-n", "--lines", type=self._nonnegative_int, default=10)
            parser.add_argument("files", nargs="*")

        options = self._parse_args("head", normalized, configure)
        if len(options.files) > 1:
            return self._error("head: at most one file is supported")
        if options.files:
            path = ws.resolve(options.files[0])
            if path is None:
                return self._error(f"Invalid path: {options.files[0]}")
            if not path.exists():
                return self._error(f"No such file: {options.files[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()[:options.lines]
        return self._ok("\n".join(lines))

    def _cmd_tail(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        normalized = [
            part
            for arg in args
            for part in (["-n", arg[1:]] if re.fullmatch(r"-\d+", arg) else [arg])
        ]

        def configure(parser):
            parser.add_argument("-n", "--lines", type=self._nonnegative_int, default=10)
            parser.add_argument("files", nargs="*")

        options = self._parse_args("tail", normalized, configure)
        if len(options.files) > 1:
            return self._error("tail: at most one file is supported")
        if options.files:
            path = ws.resolve(options.files[0])
            if path is None:
                return self._error(f"Invalid path: {options.files[0]}")
            if not path.exists():
                return self._error(f"No such file: {options.files[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()[-options.lines:] if options.lines else []
        return self._ok("\n".join(lines))

    def _cmd_wc(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-l", "--lines", action="store_true")
            parser.add_argument("-w", "--words", action="store_true")
            parser.add_argument("-c", "--bytes", action="store_true")
            parser.add_argument("-m", "--chars", action="store_true")
            parser.add_argument("files", nargs="*")

        options = self._parse_args("wc", args, configure)
        selected = [options.lines, options.words, options.bytes, options.chars]
        enabled_counts = selected if any(selected) else [True, True, True, False]
        sources: list[tuple[str, str]] = []
        if options.files:
            for name in options.files:
                path = ws.resolve(name)
                if path is None:
                    return self._error(f"Invalid path: {name}")
                if not path.exists():
                    return self._error(f"No such file: {name}")
                sources.append((name, self._read_text(path)))
        else:
            sources.append(("", stdin))

        def counts(text: str) -> list[int]:
            values = [
                len(text.splitlines()),
                len(text.split()),
                len(text.encode("utf-8")),
                len(text),
            ]
            return [
                value for value, enabled in zip(values, enabled_counts) if enabled
            ]

        output = [
            " ".join(map(str, counts(text))) + (f" {name}" if name else "")
            for name, text in sources
        ]
        if len(sources) > 1:
            totals = [sum(values) for values in zip(*(counts(text) for _, text in sources))]
            output.append(" ".join(map(str, totals)) + " total")
        return self._ok("\n".join(output))

    def _cmd_grep(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-i", "--ignore-case", action="store_true")
            parser.add_argument("-v", "--invert-match", action="store_true")
            parser.add_argument("-n", "--line-number", action="store_true")
            parser.add_argument("-c", "--count", action="store_true")
            parser.add_argument("-F", "--fixed-strings", action="store_true")
            parser.add_argument("pattern")
            parser.add_argument("files", nargs="*")

        options = self._parse_args("grep", args, configure)
        if len(options.files) > 1:
            return self._error("grep: at most one file is supported")
        if options.files:
            path = ws.resolve(options.files[0])
            if path is None:
                return self._error(f"Invalid path: {options.files[0]}")
            if not path.exists():
                return self._error(f"No such file: {options.files[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        pattern = options.pattern.casefold() if options.ignore_case else options.pattern
        matched = []
        for number, line in enumerate(text.splitlines(), 1):
            searchable = line.casefold() if options.ignore_case else line
            include = pattern in searchable
            if include == options.invert_match:
                continue
            matched.append(
                f"{number}:{line}" if options.line_number and not options.count else line
            )
        return self._ok(str(len(matched)) if options.count else "\n".join(matched))

    def _cmd_sort(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-r", "--reverse", action="store_true")
            parser.add_argument("-n", "--numeric-sort", action="store_true")
            parser.add_argument("-u", "--unique", action="store_true")
            parser.add_argument("files", nargs="*")

        options = self._parse_args("sort", args, configure)
        if len(options.files) > 1:
            return self._error("sort: at most one file is supported")
        if options.files:
            path = ws.resolve(options.files[0])
            if path is None:
                return self._error(f"Invalid path: {options.files[0]}")
            if not path.exists():
                return self._error(f"No such file: {options.files[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        lines = text.splitlines()
        if options.unique:
            lines = list(dict.fromkeys(lines))
        if options.numeric_sort:
            try:
                lines.sort(key=lambda line: float(line.strip()), reverse=options.reverse)
            except ValueError:
                return self._error("sort: non-numeric input with -n")
        else:
            lines.sort(reverse=options.reverse)
        return self._ok("\n".join(lines))

    def _cmd_cut(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-d", "--delimiter", default="\t")
            parser.add_argument("-f", "--fields", required=True)
            parser.add_argument("files", nargs="*")

        options = self._parse_args("cut", args, configure)
        if len(options.delimiter) != 1:
            return self._error("cut: delimiter must be a single character")
        if len(options.files) > 1:
            return self._error("cut: at most one file is supported")
        if options.files:
            path = ws.resolve(options.files[0])
            if path is None:
                return self._error(f"Invalid path: {options.files[0]}")
            if not path.exists():
                return self._error(f"No such file: {options.files[0]}")
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = stdin
        # 解析字段索引（1-based）
        try:
            field_indices = [int(field) - 1 for field in options.fields.split(",")]
            if any(index < 0 for index in field_indices):
                raise ValueError
        except ValueError:
            return self._error(f"Invalid field specification: {options.fields}")
        output_lines = []
        for line in text.splitlines():
            parts = line.split(options.delimiter)
            selected = [parts[i] if i < len(parts) else "" for i in field_indices]
            output_lines.append(options.delimiter.join(selected))
        return self._ok("\n".join(output_lines))

    def _cmd_cp(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-r", "-R", "--recursive", action="store_true")
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("paths", nargs="+")

        options = self._parse_args("cp", args, configure)
        if len(options.paths) != 2:
            return self._error("cp: exactly one source and one destination are supported")
        source, destination = options.paths
        src_path = ws.resolve(source)
        dst_path = ws.resolve(destination)
        if src_path is None:
            return self._error(f"Invalid source path: {source}")
        if dst_path is None:
            return self._error(f"Invalid destination path: {destination}")
        if not src_path.exists():
            return self._error(f"No such file: {source}")
        if src_path.is_dir() and not options.recursive:
            return self._error(f"cp: omitting directory {source!r}; use -r")
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=options.force)
            else:
                shutil.copy2(src_path, dst_path)
            return self._ok("")
        except Exception as e:
            return self._error(f"cp failed: {e}")

    def _cmd_mv(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("paths", nargs="+")

        options = self._parse_args("mv", args, configure)
        if len(options.paths) != 2:
            return self._error("mv: exactly one source and one destination are supported")
        source, destination = options.paths
        src_path = ws.resolve(source)
        dst_path = ws.resolve(destination)
        if src_path is None:
            return self._error(f"Invalid source path: {source}")
        if dst_path is None:
            return self._error(f"Invalid destination path: {destination}")
        if not src_path.exists():
            return self._error(f"No such file: {source}")
        if src_path == ws.path:
            return self._error("mv: refusing to move the workspace root")
        try:
            if options.force and dst_path.exists() and not dst_path.is_dir():
                dst_path.unlink()
            shutil.move(str(src_path), str(dst_path))
            return self._ok("")
        except Exception as e:
            return self._error(f"mv failed: {e}")

    def _cmd_rm(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-r", "-R", "--recursive", action="store_true")
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("paths", nargs="+")

        options = self._parse_args("rm", args, configure)
        for name in options.paths:
            path = ws.resolve(name)
            if path is None:
                return self._error(f"Invalid path: {name}")
            if path == ws.path:
                return self._error("rm: refusing to remove the workspace root")
            if not path.exists():
                if options.force:
                    continue
                return self._error(f"No such file: {name}")
            try:
                if path.is_dir():
                    if not options.recursive:
                        return self._error(f"rm: cannot remove directory {name!r} without -r")
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception as e:
                return self._error(f"rm failed: {e}")
        return self._ok("")

    def _cmd_mkdir(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-p", "--parents", action="store_true")
            parser.add_argument("directories", nargs="+")

        options = self._parse_args("mkdir", args, configure)
        for name in options.directories:
            path = ws.resolve(name)
            if path is None:
                return self._error(f"Invalid path: {name}")
            try:
                path.mkdir(parents=options.parents, exist_ok=options.parents)
            except Exception as e:
                return self._error(f"mkdir failed: {e}")
        return self._ok("")

    def _cmd_gunzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-k", "--keep", action="store_true")
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("file")

        options = self._parse_args("gunzip", args, configure)
        path = ws.resolve(options.file)
        if path is None:
            return self._error(f"Invalid path: {options.file}")
        if not path.exists():
            return self._error(f"No such file: {options.file}")
        out_path = (
            path.with_suffix("")
            if path.suffix == ".gz"
            else path.parent / (path.name + ".out")
        )
        if out_path.exists() and not options.force:
            return self._error(f"gunzip: output already exists: {out_path.name}; use -f")
        try:
            with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            if not options.keep:
                path.unlink()
            return self._ok(f"Decompressed to {out_path.name}")
        except Exception as e:
            return self._error(f"gunzip failed: {e}")

    def _cmd_gzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        def configure(parser):
            parser.add_argument("-k", "--keep", action="store_true")
            parser.add_argument("-f", "--force", action="store_true")
            parser.add_argument("file")

        options = self._parse_args("gzip", args, configure)
        path = ws.resolve(options.file)
        if path is None:
            return self._error(f"Invalid path: {options.file}")
        if not path.exists():
            return self._error(f"No such file: {options.file}")
        out_path = path.parent / (path.name + ".gz")
        if out_path.exists() and not options.force:
            return self._error(f"gzip: output already exists: {out_path.name}; use -f")
        try:
            with open(path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            if not options.keep:
                path.unlink()
            return self._ok(f"Compressed to {out_path.name}")
        except Exception as e:
            return self._error(f"gzip failed: {e}")

    def _cmd_unzip(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        import zipfile

        def configure(parser):
            parser.add_argument("-d", "--directory", default=".")
            parser.add_argument("file")

        options = self._parse_args("unzip", args, configure)
        path = ws.resolve(options.file)
        destination = ws.resolve(options.directory)
        if path is None:
            return self._error(f"Invalid path: {options.file}")
        if destination is None:
            return self._error(f"Invalid destination path: {options.directory}")
        if not path.exists():
            return self._error(f"No such file: {options.file}")
        try:
            with zipfile.ZipFile(path, "r") as zf:
                destination.mkdir(parents=True, exist_ok=True)
                for member in zf.infolist():
                    member_path = (destination / member.filename).resolve()
                    try:
                        member_path.relative_to(destination.resolve())
                    except ValueError:
                        return self._error(f"Unsafe path in archive: {member.filename}")
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        return self._error(
                            f"Symbolic links are not allowed in archives: {member.filename}"
                        )
                zf.extractall(destination)
            return self._ok(f"Extracted to {options.directory}")
        except Exception as e:
            return self._error(f"unzip failed: {e}")

    def _cmd_tar(self, args: list, ws: Workspace, stdin: str) -> Dict[str, Any]:
        import tarfile

        def configure(parser):
            parser.add_argument("-x", "--extract", action="store_true")
            parser.add_argument("-z", "--gzip", action="store_true")
            parser.add_argument("-f", "--file", required=True)
            parser.add_argument("-C", "--directory", default=".")

        options = self._parse_args("tar", args, configure)
        path = ws.resolve(options.file)
        destination = ws.resolve(options.directory)
        if path is None:
            return self._error(f"Invalid path: {options.file}")
        if destination is None:
            return self._error(f"Invalid destination path: {options.directory}")
        if not path.exists():
            return self._error(f"No such file: {options.file}")
        if not options.extract:
            return self._error("Only tar extraction (-x) is supported.")
        mode = "r:gz" if options.gzip else "r:*"
        try:
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, mode) as tf:
                tf.extractall(destination, filter="data")
            return self._ok(f"Extracted to {options.directory}")
        except Exception as e:
            return self._error(f"tar failed: {e}")
