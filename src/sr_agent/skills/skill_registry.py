# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.

import yaml
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Tuple
from dataclasses import dataclass

_logger = getLogger(f"sr_agent.{__name__}")


@dataclass
class Skill:
    name: str
    description: str
    readonly: bool
    content: str # 包括 YAML frontmatter 和 Markdown body 的整个文本内容
    path: Path


class SkillRegistry:
    """Registry for human-written symbolic-regression skills."""

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir is not None else Path(__file__).parent

    def load_skills(self) -> Dict[str, Skill]:
        """Load all valid */SKILL.md files from the skills directory."""
        skills = {}
        for skill_path in sorted(self.skills_dir.glob("*/SKILL.md")):
            metadata, content = self._load_skill_file(skill_path)
            name = str(metadata.get("name")).strip()
            description = str(metadata.get("description")).strip()
            readonly = str(metadata.get("readonly", False)).lower().strip() in [
                "true",
                "yes",
                "1",
                "on",
            ]
            if not name:
                _logger.warning(f"Skill file '{skill_path}' is missing frontmatter field: name, skip it")
                continue
            if not description:
                _logger.warning(f"Skill file '{skill_path}' is missing frontmatter field: description, skip it")
                continue
            if name != skill_path.parent.name:
                _logger.warning(
                    f"Skill name {name:!r} in {skill_path:!r} not match directory name "
                    f"{skill_path.parent.name!r}, use {name:!r} as the skill name"
                )
            skills[name] = Skill(
                name=name,
                description=description.strip(),
                readonly=readonly,
                content=content,
                path=skill_path,
            )
        return skills

    def get_skills(self, name: str) -> Skill:
        """Get a skill by name."""
        if name in (skills := self.load_skills()):
            return skills[name]
        else:
            available_skills = ", ".join(skills.keys())
            raise ValueError(f"Skill '{name}' not found. Available skills: {available_skills}")

    def _load_skill_file(self, path: Path) -> Tuple[dict[str, Any], str]:
        text = path.read_text(encoding="utf-8").strip()

        if not text.startswith("---\n"):
            raise ValueError(f"Skill file '{path}' must start with YAML frontmatter")
        try:
            _, frontmatter, _ = text.split("---\n", 2)
            frontmatter = frontmatter.strip()
        except ValueError:
            raise ValueError(f"Skill file '{path}' has invalid YAML frontmatter")
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            metadata = self._parse_simple_frontmatter(frontmatter)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Failed to parse frontmatter in skill file '{path}', "
                f"expected a YAML mapping but got {type(metadata).__name__}"
            )
        return metadata, text

    def _parse_simple_frontmatter(self, frontmatter: str) -> dict[str, Any]:
        metadata = {}
        for line in frontmatter.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip(' "\'')
            metadata[key] = value
        return metadata
