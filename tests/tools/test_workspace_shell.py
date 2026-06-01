# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""WorkspaceShellTool 的测试。

重点覆盖：
- Workspace 路径安全机制（逃逸防护）
- Shell 命令白名单与拒绝机制
- 各命令的基本功能
- Python 脚本执行
- 管道功能
"""

import tempfile
from pathlib import Path

from sr_agent.tools.workspace_shell import Workspace, WorkspaceShellTool


class TestWorkspace:
    """测试 Workspace 工作区管理器。"""

    def setup_method(self):
        self.ws = Workspace()

    def teardown_method(self):
        self.ws.cleanup()

    def test_workspace_creates_temp_dir(self):
        """工作区正确创建临时目录。"""
        assert self.ws.path.exists()
        assert self.ws.path.is_dir()

    def test_resolve_valid_relative_path(self):
        """合法相对路径正确解析。"""
        p = self.ws.resolve("test.txt")
        assert p is not None
        assert p.parent == self.ws.path

    def test_resolve_nested_relative_path(self):
        """嵌套相对路径正确解析。"""
        subdir = self.ws.path / "subdir"
        subdir.mkdir()
        p = self.ws.resolve("subdir/file.txt")
        assert p is not None
        assert str(self.ws.path) in str(p)

    def test_resolve_empty_returns_root(self):
        """空路径返回工作区根目录。"""
        p = self.ws.resolve("")
        assert p == self.ws.path

    def test_resolve_rejects_absolute_path(self):
        """绝对路径被拒绝。"""
        assert self.ws.resolve("/etc/passwd") is None
        assert self.ws.resolve("/tmp/something") is None
        # Windows 驱动器号风格也被拒绝（跨平台防御）
        assert self.ws.resolve("C:/Windows/System32") is None

    def test_resolve_rejects_parent_traversal(self):
        """路径逃逸（..）被拒绝。"""
        assert self.ws.resolve("..") is None
        assert self.ws.resolve("../secret") is None
        assert self.ws.resolve("subdir/../../etc/passwd") is None
        assert self.ws.resolve("../../../etc/passwd") is None

    def test_resolve_rejects_dot_dot_in_middle(self):
        """中间包含 .. 的路径逃逸被拒绝。"""
        assert self.ws.resolve("a/b/../../..") is None

    def test_link_file(self):
        """文件正确链接到工作区。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.write("a,b\n1,2\n")
        tmp.close()
        try:
            ws = Workspace(workspace_files=[tmp.name])
            linked = ws.path / Path(tmp.name).name
            assert linked.exists()
            assert "a,b" in linked.read_text()
            ws.cleanup()
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_link_directory(self):
        """目录正确链接到工作区。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "file.txt").write_text("hello")
            ws = Workspace(workspace_files=[tmp_dir])
            linked_dir = ws.path / Path(tmp_dir).name
            assert linked_dir.exists()
            assert (linked_dir / "file.txt").read_text() == "hello"
            ws.cleanup()

    def test_cleanup_removes_workspace(self):
        """cleanup 正确删除工作区目录。"""
        ws = Workspace()
        path = ws.path
        assert path.exists()
        ws.cleanup()
        assert not path.exists()

    def test_context_manager(self):
        """上下文管理器在退出时清理工作区。"""
        with Workspace() as ws:
            path = ws.path
            assert path.exists()
        assert not path.exists()


class TestWorkspaceShellTool:
    """测试 WorkspaceShellTool 命令执行。"""

    def setup_method(self):
        self.ws = Workspace()
        # 创建测试文件
        (self.ws.path / "data.csv").write_text("name,value\nalice,10\nbob,20\ncharlie,30\n")
        (self.ws.path / "numbers.txt").write_text("5\n3\n8\n1\n4\n")
        (self.ws.path / "subdir").mkdir()
        (self.ws.path / "subdir" / "nested.txt").write_text("nested content")
        self.tool = WorkspaceShellTool(workspace=self.ws)

    def teardown_method(self):
        self.ws.cleanup()

    # ─── 元数据 ───

    def test_metadata(self):
        """工具元数据正确。"""
        assert WorkspaceShellTool.metadata.name == "workspace_shell"
        assert WorkspaceShellTool.metadata.description is not None

    # ─── 命令白名单 ───

    def test_rejects_unknown_command(self):
        """非白名单命令被拒绝。"""
        result = self.tool.execute("wget http://example.com")
        assert result.get("success") is False
        assert "not allowed" in result.get("error", "")

    def test_rejects_curl(self):
        """curl 被拒绝。"""
        result = self.tool.execute("curl http://example.com")
        assert result.get("success") is False

    def test_rejects_chmod(self):
        """chmod 被拒绝。"""
        result = self.tool.execute("chmod 777 data.csv")
        assert result.get("success") is False

    def test_rejects_bash(self):
        """bash 被拒绝。"""
        result = self.tool.execute("bash -c 'echo pwned'")
        assert result.get("success") is False

    # ─── 路径逃逸防护（仅使用无害命令 ls/cat） ───

    def test_ls_rejects_absolute_path(self):
        """ls 绝对路径被拒绝。"""
        result = self.tool.execute("ls /etc")
        assert result.get("success") is False
        assert "Invalid path" in result.get("error", "")

    def test_ls_rejects_parent_traversal(self):
        """ls 路径逃逸（..）被拒绝。"""
        result = self.tool.execute("ls ..")
        assert result.get("success") is False
        assert "Invalid path" in result.get("error", "")

    def test_ls_rejects_deep_traversal(self):
        """ls 深层路径逃逸被拒绝。"""
        result = self.tool.execute("ls ../../../")
        assert result.get("success") is False

    def test_cat_rejects_absolute_path(self):
        """cat 绝对路径被拒绝。"""
        result = self.tool.execute("cat /etc/passwd")
        assert result.get("success") is False
        assert "Invalid path" in result.get("error", "")

    def test_cat_rejects_parent_traversal(self):
        """cat 路径逃逸被拒绝。"""
        result = self.tool.execute("cat ../../../etc/passwd")
        assert result.get("success") is False
        assert "Invalid path" in result.get("error", "")

    def test_head_rejects_traversal(self):
        """head 路径逃逸被拒绝。"""
        result = self.tool.execute("head ../../etc/passwd")
        assert result.get("success") is False

    def test_grep_rejects_traversal(self):
        """grep 路径逃逸被拒绝。"""
        result = self.tool.execute("grep root ../../../etc/passwd")
        assert result.get("success") is False

    # ─── ls 命令 ───

    def test_ls_workspace_root(self):
        """ls 列出工作区根目录。"""
        result = self.tool.execute("ls")
        assert result.get("success") is True
        stdout = result["stdout"]
        assert "data.csv" in stdout
        assert "numbers.txt" in stdout
        assert "subdir/" in stdout

    def test_ls_subdirectory(self):
        """ls 列出子目录。"""
        result = self.tool.execute("ls subdir")
        assert result.get("success") is True
        assert "nested.txt" in result["stdout"]

    def test_ls_nonexistent(self):
        """ls 不存在的路径报错。"""
        result = self.tool.execute("ls nonexistent")
        assert result.get("success") is False
        assert "No such file" in result.get("error", "")

    # ─── cat 命令 ───

    def test_cat_file(self):
        """cat 正确读取文件内容。"""
        result = self.tool.execute("cat data.csv")
        assert result.get("success") is True
        assert "alice,10" in result["stdout"]
        assert "bob,20" in result["stdout"]

    def test_cat_nested_file(self):
        """cat 读取子目录中的文件。"""
        result = self.tool.execute("cat subdir/nested.txt")
        assert result.get("success") is True
        assert "nested content" in result["stdout"]

    def test_cat_nonexistent(self):
        """cat 不存在的文件报错。"""
        result = self.tool.execute("cat missing.txt")
        assert result.get("success") is False

    # ─── head/tail 命令 ───

    def test_head_default(self):
        """head 默认显示前 10 行。"""
        result = self.tool.execute("head data.csv")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert len(lines) == 4  # 文件只有 4 行

    def test_head_with_n(self):
        """head -n N 显示指定行数。"""
        result = self.tool.execute("head -n 2 data.csv")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert len(lines) == 2
        assert "name,value" in lines[0]

    def test_tail_with_n(self):
        """tail -n N 显示末尾行数。"""
        result = self.tool.execute("tail -n 2 data.csv")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert len(lines) == 2
        assert "charlie,30" in lines[-1]

    # ─── wc 命令 ───

    def test_wc_file(self):
        """wc 正确统计行数、词数、字符数。"""
        result = self.tool.execute("wc data.csv")
        assert result.get("success") is True
        parts = result["stdout"].strip().split()
        assert int(parts[0]) == 4  # 4 行

    # ─── grep 命令 ───

    def test_grep_pattern(self):
        """grep 过滤包含模式的行。"""
        result = self.tool.execute("grep bob data.csv")
        assert result.get("success") is True
        assert "bob,20" in result["stdout"]
        assert "alice" not in result["stdout"]

    def test_grep_no_match(self):
        """grep 无匹配时返回空。"""
        result = self.tool.execute("grep zebra data.csv")
        assert result.get("success") is True
        assert result["stdout"] == ""

    # ─── sort 命令 ───

    def test_sort_file(self):
        """sort 按字母排序。"""
        result = self.tool.execute("sort numbers.txt")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert lines[0] == "1"
        assert lines[-1] == "8"

    # ─── cut 命令 ───

    def test_cut_fields(self):
        """cut 提取指定列。"""
        result = self.tool.execute("cut -d , -f 1 data.csv")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert "name" in lines[0]
        assert "alice" in lines[1]
        assert "10" not in result["stdout"]

    # ─── 管道 ───

    def test_pipe_cat_head(self):
        """管道：cat | head。"""
        result = self.tool.execute("cat data.csv | head -2")
        assert result.get("success") is True
        lines = result["stdout"].strip().splitlines()
        assert len(lines) == 2

    def test_pipe_cat_grep(self):
        """管道：cat | grep。"""
        result = self.tool.execute("cat data.csv | grep alice")
        assert result.get("success") is True
        assert "alice" in result["stdout"]
        assert "bob" not in result["stdout"]

    def test_pipe_cat_wc(self):
        """管道：cat | wc。"""
        result = self.tool.execute("cat data.csv | wc")
        assert result.get("success") is True
        parts = result["stdout"].strip().split()
        assert int(parts[0]) == 4

    def test_output_is_limited(self):
        """长输出会被截断。"""
        (self.ws.path / "large.txt").write_text("x" * 5000)
        result = self.tool.execute("cat large.txt", output_limit_bytes=1024)
        assert result.get("success") is True
        assert result["stdout"].endswith("...[truncated]")
        assert len(result["stdout"]) == 1024

    def test_pipe_intermediate_output_is_limited(self):
        """管道中间结果也会被截断，避免传递过长输出。"""
        (self.ws.path / "large.txt").write_text("x" * 5000)
        result = self.tool.execute("cat large.txt | wc", output_limit_bytes=1024)
        assert result.get("success") is True
        assert result["stdout"].strip().split() == ["1", "1", "1024"]

    # ─── 文件操作命令 ───

    def test_mkdir(self):
        """mkdir 创建目录。"""
        result = self.tool.execute("mkdir newdir")
        assert result.get("success") is True
        assert (self.ws.path / "newdir").is_dir()

    def test_cp_file(self):
        """cp 复制文件。"""
        result = self.tool.execute("cp data.csv data_copy.csv")
        assert result.get("success") is True
        assert (self.ws.path / "data_copy.csv").exists()
        assert (self.ws.path / "data_copy.csv").read_text() == (self.ws.path / "data.csv").read_text()

    def test_mv_file(self):
        """mv 移动/重命名文件。"""
        (self.ws.path / "temp.txt").write_text("temp content")
        result = self.tool.execute("mv temp.txt renamed.txt")
        assert result.get("success") is True
        assert not (self.ws.path / "temp.txt").exists()
        assert (self.ws.path / "renamed.txt").read_text() == "temp content"

    def test_cp_rejects_traversal(self):
        """cp 目标路径逃逸被拒绝。"""
        result = self.tool.execute("cp data.csv ../../escaped.csv")
        assert result.get("success") is False
        assert "Invalid" in result.get("error", "")

    # ─── 压缩/解压 ───

    def test_gzip_gunzip(self):
        """gzip 和 gunzip 压缩/解压。"""
        (self.ws.path / "compress_me.txt").write_text("compress this content")
        result = self.tool.execute("gzip compress_me.txt")
        assert result.get("success") is True
        assert (self.ws.path / "compress_me.txt.gz").exists()
        assert not (self.ws.path / "compress_me.txt").exists()

        result = self.tool.execute("gunzip compress_me.txt.gz")
        assert result.get("success") is True
        assert (self.ws.path / "compress_me.txt").exists()
        assert (self.ws.path / "compress_me.txt").read_text() == "compress this content"

    # ─── format_result_dict ───

    def test_format_result_success(self):
        """成功结果格式化输出 stdout。"""
        result = {"success": True, "stdout": "file1\nfile2\n", "stderr": "", "exit_code": 0}
        formatted = WorkspaceShellTool.format_result_dict(result)
        assert "file1" in formatted

    def test_format_result_error(self):
        """失败结果格式化输出 error。"""
        result = {"success": False, "stdout": "", "stderr": "", "error": "Path invalid", "exit_code": 1}
        formatted = WorkspaceShellTool.format_result_dict(result)
        assert "Path invalid" in formatted

    # ─── __call__ 接口 ───

    def test_tool_call_interface(self):
        """通过 __call__ 接口调用返回 ToolCallResult。"""
        tool_result = self.tool("ls")
        assert tool_result.ok is True
        assert "data.csv" in tool_result.result["stdout"]

