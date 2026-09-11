# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from typing import Any, Dict
from xml.sax.saxutils import escape
from ..skills import SkillRegistry
from .base_tool import BaseTool, ToolMetadata

def _format_skills_description() -> str:
    skills = SkillRegistry().load_skills()
    skill_blocks = []
    for skill in skills.values():
        skill_blocks.append(
            f"  <skill>\n"
            f"    <name>{escape(skill.name)}</name>\n"
            f"    <description>\n"
            f"      {escape(skill.description)}\n"
            f"    </description>\n"
            f"  </skill>"
        )
    return (
        "Read the content of a skill by its name.\n\n"
        "Skills are reusable human-written instructions. Use this tool when the "
        "current task matches one of the skill descriptions below.\n\n"
        "Available skills:\n\n"
        "<skills>\n"
        + "\n\n".join(skill_blocks)
        + "\n</skills>"
    )

@BaseTool.register("read_skill")
class ReadSkill(BaseTool):
    metadata = ToolMetadata(
        name="read_skill",
        description=_format_skills_description(),
    )

    def execute(self, name: str, file_path: str = "", show_tree: bool = False) -> Dict[str, Any]:
        """Inspect a skill's instructions, directory structure, or a file.

        Args:
            name: The exact skill name to inspect.
            file_path: Optional path relative to the skill directory, such as
                ``tool.py`` or ``references/example.md``. Empty reads SKILL.md.
            show_tree: Whether to include all files and subdirectories in the skill.
        """
        skills_dir = self.context.get("skills_dir", None)
        registry = SkillRegistry(skills_dir)
        skill = registry.get_skills(name)
        skill_dir = skill.path.parent.resolve()
        result = {}
        if file_path.strip():
            requested = (skill_dir / file_path.strip()).resolve()
            try:
                requested.relative_to(skill_dir)
            except ValueError:
                raise ValueError("file_path must stay inside the skill directory.")
            if not requested.is_file():
                raise ValueError(f"Skill file not found: {file_path}")
            result["file_path"] = requested.relative_to(skill_dir).as_posix()
            result["file_content"] = requested.read_text(encoding="utf-8")
            result["content"] = (
                f'<skill_content name="{escape(skill.name)}" file="{escape(result["file_path"])}">\n'
                f'{result["file_content"]}\n</skill_content>'
            )
        else:
            result["content"] = f'<skill_content name="{escape(skill.name)}">\n{skill.content}\n</skill_content>'
        if show_tree:
            result["tree"] = [
                (path.relative_to(skill_dir).as_posix() + ("/" if path.is_dir() else ""))
                for path in sorted(skill_dir.rglob("*"))
                if "__pycache__" not in path.parts
                and not (path.is_file() and path.suffix in {".pyc", ".pyo"})
            ]
        return result

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        text = [result["content"]]
        if "tree" in result:
            text.extend(["", "Skill directory tree:", *[f"- {path}" for path in result["tree"]]])
        return "\n".join(text)
