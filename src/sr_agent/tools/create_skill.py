# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import re
import yaml
from pathlib import Path
from typing import Any, Dict

from ..skills import SkillRegistry
from .base_tool import BaseTool, ToolMetadata

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@BaseTool.register("create_skill")
class CreateSkill(BaseTool):
    metadata = ToolMetadata(name="create_skill")

    def execute(
        self,
        name: str,
        description: str,
        content: str,
        readonly: bool = False,
    ) -> Dict[str, Any]:
        """Create a new skill.
        Use this only after a reusable lesson, strategy, or best practice has been
        identified and supported by concrete results from the current or recent work.
        You do not need to find a perfect formula or reach MSE = 0.0 before creating
        a skill; create one when an attempt reveals a useful reusable tactic.
        Do not create a skill for a one-off dataset detail, a final formula, or an
        unverified guess. Good skill content explains when to use the lesson, what
        evidence supports it, concrete steps to follow, and pitfalls to avoid. The
        tool refuses to overwrite existing skills.

        Args:
            name: Skill name. Must use lowercase letters, digits, and hyphens.
            description: Brief guidance for when an LLM should read this skill.
            content: Markdown body for SKILL.md, excluding YAML frontmatter.
            readonly: Whether the created skill should be marked read-only.
        """
        if not _SKILL_NAME_PATTERN.fullmatch(name := name.strip()):
            raise ValueError(
                "Skill name must use lowercase letters, digits, and hyphens "
                "and cannot start or end with a hyphen."
            )
        if not (description := description.strip()):
            raise ValueError("Skill description cannot be empty.")
        if content.strip().startswith("---"):
            _, _, content = content.split("---\n", 2) # 扔掉可能包含的 YAML frontmatter，避免重复
        if not (content := content.strip()):
            raise ValueError("Skill content cannot be empty.")

        skills_dir = self.context.get("skills_dir") # 用来测试时使用，一般用不到
        registry = SkillRegistry(skills_dir)
        if name in registry.load_skills():
            raise ValueError(f"Skill '{name}' already exists. Choose a different name.")
        elif (skill_dir := registry.skills_dir / name).exists():
            raise ValueError(f"Skill directory already exists: {skill_dir}. Choose a different name.")
        else:
            skill_dir.mkdir(parents=True, exist_ok=False)

        frontmatter = {"name": name, "description": description, "readonly": readonly}
        text = (
            f"---\n"
            f"{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)}"
            f"---\n\n"
            f"{content}\n"
        )
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(text, encoding="utf-8")
        return {
            "success": True,
            "name": name,
            "path": str(skill_path),
            "message": f"Skill created: {name}",
        }
