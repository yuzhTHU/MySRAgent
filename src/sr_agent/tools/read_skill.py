# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from pathlib import Path
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

    def execute(self, name: str) -> str:
        """Read the content of a skill by its name.

        Args:
            name: The exact skill name to read.
        """
        skills_dir = self.context.get("skills_dir", None)
        registry = SkillRegistry(skills_dir)
        skill = registry.get_skills(name)
        return (
            f'<skill_content name="{escape(skill.name)}">\n'
            f"{skill.content}\n"
            f"</skill_content>"
        )

