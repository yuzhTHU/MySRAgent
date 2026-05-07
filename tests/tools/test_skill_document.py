# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""SkillDocumentTool 的单元测试。"""

import pytest
import os
from pathlib import Path

from sr_agent.tools.skill_document import SkillDocumentTool


class TestSkillDocumentTool:
    """测试 SkillDocumentTool 的正确性。"""

    def setup_method(self):
        """每个测试方法前执行。"""
        self.tool = SkillDocumentTool()

    def test_list_skills(self):
        """测试列出所有 Skill。"""
        result = self.tool.execute(operation="list")

        assert result["success"] is True
        assert result["operation"] == "list"
        assert "skills" in result
        assert "base_path" in result

        # 检查是否找到 symbolic-regression skill
        skill_names = [s["name"] for s in result["skills"]]
        assert "symbolic-regression" in skill_names

    def test_list_skills_structure(self):
        """测试列出的 Skill 结构。"""
        result = self.tool.execute(operation="list")

        skill = next(s for s in result["skills"] if s["name"] == "symbolic-regression")

        # 检查 Skill 信息结构
        assert "name" in skill
        assert "description" in skill
        assert "files" in skill

        # 检查文件列表
        file_paths = [f["path"] for f in skill["files"]]
        assert "SKILL.md" in file_paths
        assert "gotchas.md" in file_paths
        assert "patterns/linear.yaml" in file_paths
        assert "references/operators.md" in file_paths

    def test_read_skill_files(self):
        """测试列出 Skill 文件。"""
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
        )

        assert result["success"] is True
        assert result["operation"] == "read"
        assert result["skill_name"] == "symbolic-regression"
        assert "files" in result
        assert len(result["files"]) > 0

    def test_read_skill_file_content(self):
        """测试读取 Skill 文件内容。"""
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
            file_path="SKILL.md",
        )

        assert result["success"] is True
        assert result["operation"] == "read"
        assert result["skill_name"] == "symbolic-regression"
        assert result["file_path"] == "SKILL.md"
        assert "content" in result
        assert len(result["content"]) > 0
        assert "Symbolic Regression" in result["content"]

    def test_read_subdirectory_file(self):
        """测试读取子目录中的文件。"""
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
            file_path="patterns/linear.yaml",
        )

        assert result["success"] is True
        assert "content" in result
        assert "线性" in result["content"] or "linear" in result["content"].lower()

    def test_read_reference_file(self):
        """测试读取参考文档。"""
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
            file_path="references/operators.md",
        )

        assert result["success"] is True
        assert len(result["content"]) > 0
        assert "nd2py" in result["content"]

    def test_invalid_skill_name(self):
        """测试无效 Skill 名称。"""
        result = self.tool.execute(
            operation="read",
            skill_name="nonexistent-skill",
        )

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_invalid_file_path(self):
        """测试无效文件路径。"""
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
            file_path="nonexistent.md",
        )

        assert result["success"] is False
        assert "error" in result

    def test_path_traversal_protection(self):
        """测试路径遍历攻击防护。"""
        # 尝试访问父目录
        result = self.tool.execute(
            operation="read",
            skill_name="symbolic-regression",
            file_path="../../../etc/passwd",
        )

        assert result["success"] is False
        assert "error" in result

    def test_path_traversal_in_skill_name(self):
        """测试 Skill 名称中的路径遍历防护。"""
        result = self.tool.execute(
            operation="read",
            skill_name="../tools",
            file_path="base_tool.py",
        )

        assert result["success"] is False

    def test_missing_skill_name_for_read(self):
        """测试 read 操作缺少 skill_name 参数。"""
        result = self.tool.execute(operation="read")

        assert result["success"] is False
        assert "error" in result
        assert "skill_name" in result["error"]

    def test_invalid_operation(self):
        """测试无效操作类型。"""
        result = self.tool.execute(operation="invalid")

        assert result["success"] is False
        assert "error" in result
        assert "Unknown operation" in result["error"]

    def test_metadata_exists(self):
        """测试元数据存在。"""
        assert self.tool.metadata is not None
        assert self.tool.metadata.name == "skill_document"
        assert "list" in self.tool.metadata.description.lower()
        assert "read" in self.tool.metadata.description.lower()

    def test_file_type_guessing(self):
        """测试文件类型推断。"""
        assert self.tool._guess_file_type(Path("test.md")) == "documentation"
        assert self.tool._guess_file_type(Path("config.yaml")) == "config"
        assert self.tool._guess_file_type(Path("script.py")) == "script"
        assert self.tool._guess_file_type(Path("data.csv")) == "data"
        assert self.tool._guess_file_type(Path("unknown.xyz")) == "other"

    def test_hidden_files_skipped(self):
        """测试隐藏文件被跳过。"""
        # .gitignore 等隐藏文件不应出现在列表中
        result = self.tool.execute(operation="list")
        for skill in result["skills"]:
            file_paths = [f["path"] for f in skill["files"]]
            for path in file_paths:
                assert not Path(path).name.startswith(".")
