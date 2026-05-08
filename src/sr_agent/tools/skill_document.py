# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Skill 文档浏览工具。

允许 LLM 浏览和读取 skills 文件夹中的文档，获取专家知识和最佳实践指导。
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from .base_tool import BaseTool, ToolMetadata

# @BaseTool.register('skill_document') # 不注册这个工具，因为它用处不大
class SkillDocumentTool(BaseTool):
    """Browse and read Skill documentation files.

    This tool allows LLM to explore available skills and read their
    documentation for guidance on specific topics.

    Skills are stored in the `src/sr_agent/skills/` directory, with each
    skill being a folder containing markdown files, yaml configs, scripts,
    and other resources.

    Use cases:
    - Discovering available skills and their purposes
    - Reading best practices and methodology guidance
    - Accessing formula patterns and templates
    - Looking up operator references and API documentation

    Operations:
    - list: Show all available skills and their file structure
    - read: Read a specific file from a skill folder
    """

    metadata = ToolMetadata(
        name="skill_document",
        description="Browse Skill documentation. Use 'list' operation to see available skills and their file structures, 'read' operation to get specific guidance from a skill document. Skills contain expert knowledge, best practices, and formula patterns for symbolic regression.",
    )

    # Skill 文件夹的基础路径
    SKILL_BASE = Path(__file__).parent.parent / "skills"

    def execute(
        self,
        operation: Literal["list", "read"],
        skill_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute skill document operation.

        Args:
            operation: Operation type - "list" to show skills, "read" to get file content
            skill_name: Skill folder name (required for "read" operation)
            file_path: Relative file path within skill folder (required for "read" operation)

        Returns:
            Dictionary containing:
            - success: Boolean indicating whether operation succeeded
            - operation: The operation that was performed
            - skills: List of available skills (for "list" operation)
            - content: File content (for "read" operation)
            - files: List of files in a skill (for "read" with file_path=None)
            - error: Error message if failed
        """
        try:
            if operation == "list":
                return self._list_skills()
            elif operation == "read":
                return self._read_skill(skill_name, file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}. Use 'list' or 'read'.",
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def _list_skills(self) -> Dict[str, Any]:
        """List all available skills and their directory structures.

        Returns:
            Dictionary containing:
            - success: True
            - operation: "list"
            - skills: List of skill information
            - base_path: The base path for skills
        """
        skills = []

        if not self.SKILL_BASE.exists():
            return {
                "success": True,
                "operation": "list",
                "skills": [],
                "base_path": str(self.SKILL_BASE),
                "message": "Skills directory does not exist yet. Create skill folders in src/sr_agent/skills/",
            }

        for skill_dir in sorted(self.SKILL_BASE.iterdir()):
            if not skill_dir.is_dir():
                continue

            # 跳过隐藏目录
            if skill_dir.name.startswith("."):
                continue

            skill_info = {
                "name": skill_dir.name,
                "description": self._get_skill_description(skill_dir),
                "files": self._list_skill_files(skill_dir),
            }
            skills.append(skill_info)

        return {
            "success": True,
            "operation": "list",
            "skills": skills,
            "base_path": str(self.SKILL_BASE),
        }

    def _read_skill(
        self,
        skill_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read a specific skill file or list files in a skill.

        Args:
            skill_name: Skill folder name
            file_path: Relative file path within skill folder.
                       If None, lists all files in the skill.

        Returns:
            Dictionary containing:
            - success: Boolean
            - operation: "read"
            - skill_name: The skill name
            - file_path: The file path that was read
            - content: File content (if file_path was provided)
            - files: List of files (if file_path was None)
            - error: Error message if failed
        """
        if not skill_name:
            return {
                "success": False,
                "error": "skill_name is required for 'read' operation",
            }

        skill_dir = self._safe_skill_dir(skill_name)

        # 如果只提供了 skill_name，返回文件列表
        if file_path is None:
            return {
                "success": True,
                "operation": "read",
                "skill_name": skill_name,
                "files": self._list_skill_files(skill_dir),
                "message": "No file_path specified. Use file_path to read a specific file.",
            }

        # 读取具体文件
        target_path = self._safe_path(skill_dir, file_path)

        # 读取文件内容
        try:
            content = target_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "operation": "read",
                "skill_name": skill_name,
                "file_path": file_path,
                "content": content,
            }
        except UnicodeDecodeError:
            # 二进制文件无法读取
            return {
                "success": False,
                "error": f"Cannot read binary file: {file_path}",
            }

    def _safe_skill_dir(self, skill_name: str) -> Path:
        """Get safe skill directory path with validation.

        Args:
            skill_name: Name of the skill folder

        Returns:
            Validated Path object for the skill directory

        Raises:
            ValueError: If skill directory doesn't exist or path traversal detected
        """
        # 防止路径遍历攻击
        if ".." in skill_name or skill_name.startswith("/"):
            raise ValueError(f"Invalid skill name: {skill_name}")

        skill_dir = self.SKILL_BASE / skill_name

        if not skill_dir.exists():
            available = [d.name for d in self.SKILL_BASE.iterdir() if d.is_dir()]
            raise ValueError(
                f"Skill '{skill_name}' not found. Available skills: {available}"
            )

        if not skill_dir.is_dir():
            raise ValueError(f"'{skill_name}' is not a directory")

        return skill_dir.resolve()

    def _safe_path(self, skill_dir: Path, file_path: str) -> Path:
        """Ensure file path doesn't escape skill directory.

        Args:
            skill_dir: Base skill directory (resolved)
            file_path: Relative file path within skill

        Returns:
            Validated Path object

        Raises:
            ValueError: If path traversal detected or file not found
        """
        # 规范化路径
        file_path = file_path.replace("\\", "/")

        # 防止路径遍历
        if ".." in file_path:
            raise ValueError(f"Path traversal detected: {file_path}")

        # 计算完整路径
        target_path = (skill_dir / file_path).resolve()

        # 确保路径在 skill 目录内
        if not str(target_path).startswith(str(skill_dir)):
            raise ValueError(f"Path traversal detected: {file_path}")

        if not target_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path} in skill '{skill_dir.name}'"
            )

        return target_path

    def _list_skill_files(self, skill_dir: Path) -> List[Dict[str, str]]:
        """List all files in a skill directory recursively.

        Args:
            skill_dir: Skill directory path

        Returns:
            List of file information dictionaries
        """
        files = []

        for root, dirs, filenames in os.walk(skill_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            rel_root = Path(root).relative_to(skill_dir)

            for filename in sorted(filenames):
                # 跳过隐藏文件和字节码文件
                if filename.startswith(".") or filename.endswith(".pyc"):
                    continue

                rel_path = rel_root / filename if str(rel_root) != "." else Path(filename)
                file_info = {
                    "path": str(rel_path).replace("\\", "/"),
                    "type": self._guess_file_type(rel_path),
                    "size": (skill_dir / rel_path).stat().st_size,
                }
                files.append(file_info)

        return files

    def _get_skill_description(self, skill_dir: Path) -> str:
        """Get skill description from SKILL.md or README.md.

        Args:
            skill_dir: Skill directory path

        Returns:
            Description string extracted from the first 200 chars of SKILL.md
        """
        for desc_file in ["SKILL.md", "README.md"]:
            desc_path = skill_dir / desc_file
            if desc_path.exists():
                try:
                    content = desc_path.read_text(encoding="utf-8")
                    # 提取第一段作为描述
                    first_para = content.split("\n\n")[0]
                    # 去除 markdown 标题标记
                    first_para = first_para.lstrip("#").strip()
                    # 截断到 200 字符
                    if len(first_para) > 200:
                        first_para = first_para[:197] + "..."
                    return first_para
                except Exception:
                    pass

        return "No description available"

    def _guess_file_type(self, file_path: Path) -> str:
        """Guess file type based on extension.

        Args:
            file_path: Relative file path

        Returns:
            File type category string
        """
        ext = file_path.suffix.lower()

        type_map = {
            ".md": "documentation",
            ".txt": "documentation",
            ".yaml": "config",
            ".yml": "config",
            ".json": "config",
            ".py": "script",
            ".sh": "script",
            ".bash": "script",
            ".sql": "script",
            ".csv": "data",
            ".sqlite": "data",
            ".db": "data",
        }

        return type_map.get(ext, "other")
