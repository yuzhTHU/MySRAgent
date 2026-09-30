# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from typing import Any, Dict
from dataclasses import replace
from xml.sax.saxutils import escape
from ..skills import Skill, SkillManager
from .base_tool import BaseTool, ToolMetadata


def _format_description(skills: Dict[str, Skill]) -> str:
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
        "<skills>\n" + "\n\n".join(skill_blocks) + "\n</skills>"
    )


@BaseTool.register("read_skill")
class ReadSkill(BaseTool):
    default_skill_manager = SkillManager()
    metadata = ToolMetadata(
        name="read_skill",
        description=_format_description(default_skill_manager.load_skills()),
    )

    def __init__(self, **context):
        super().__init__(**context)
        self.skill_manager = context.get("skill_manager") or self.default_skill_manager
        description = _format_description(self.skill_manager.load_skills())
        self.metadata = replace(type(self).metadata, description=description)

    def execute(self, name: str, file_path: str = "", show_tree: bool = False) -> Dict[str, Any]:
        """Inspect a skill's instructions, directory structure, or a file.

        Args:
            name: The exact skill name to inspect.
            file_path: Optional path relative to the skill directory, such as
                ``tool.py`` or ``references/example.md``. Empty reads SKILL.md.
            show_tree: Whether to include all files and subdirectories in the skill.
        """
        skill = self.skill_manager.get_skill(name)
        result: dict[str, Any] = {}
        if file_path.strip():
            relative_path = file_path.strip()
            file_content = self.skill_manager.read_skill(name, relative_path)
            result.update(file_path=relative_path, file_content=file_content)
            result["content"] = (
                f'<skill_content name="{escape(skill.name)}" file="{escape(relative_path)}">\n'
                f"{file_content}\n</skill_content>"
            )
        else:
            content = self.skill_manager.read_skill(name)
            result["content"] = f'<skill_content name="{escape(skill.name)}">\n{content}\n</skill_content>'
        if show_tree:
            result["tree"] = self.skill_manager.get_skill_tree(name)
        return result

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        text = [result["content"]]
        if "tree" in result:
            text.extend(["", "Skill directory tree:", *[f"- {path}" for path in result["tree"]]])
        return "\n".join(text)
