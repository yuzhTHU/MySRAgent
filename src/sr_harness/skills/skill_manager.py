# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable

import yaml

_logger = getLogger(f"sr_agent.{__name__}")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    readonly: bool
    skill_directory: Path


class SkillManager:
    """Manage built-in, runtime, and custom skills through one interface."""

    def __init__(
        self,
        built_in_directory: str | Path | None = None,
        custom_directory: str | Path | None = None,
    ):
        package_directory = Path(__file__).parent
        self.skill_directories: dict[str, Path] = {
            "built-in": Path(built_in_directory or package_directory),
            "runtime": Path(tempfile.mkdtemp(prefix="sr_agent_runtime_skills_")),
            "custom": Path(custom_directory or package_directory / "custom"),
        }
        self.skill_directories["custom"].mkdir(parents=True, exist_ok=True)

    def register_tool_docs(self, tool_cls_list: Iterable[type]) -> None:
        """Materialize documentation from enabled tools as read-only runtime skills."""
        for tool_cls in tool_cls_list:
            doc = tool_cls.get_doc()
            if doc is None:
                continue
            if not isinstance(doc, dict):
                raise TypeError(f"{tool_cls.__name__}.get_doc() must return a dict or None.")
            if set(doc) != {"name", "description", "content"}:
                raise ValueError(
                    f"{tool_cls.__name__}.get_doc() must return exactly name, description, and content."
                )
            if not all(isinstance(doc[key], str) and doc[key].strip() for key in doc):
                raise ValueError(f"{tool_cls.__name__}.get_doc() values must be non-empty strings.")
            name = self._validate_name(doc["name"])
            if name in self.load_skills():
                raise ValueError(f"Skill {name!r} is already registered.")
            content = self._format_skill_file(
                name=name,
                description=doc["description"].strip(),
                readonly=True,
                body=doc["content"].strip(),
            )
            skill_dir = self.skill_directories["runtime"] / name
            skill_dir.mkdir(parents=True, exist_ok=False)
            (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    def discover_tool_skills(self) -> list[Skill]:
        """Return all registered skills that contain a ``tool.py`` file."""
        return [
            skill
            for skill in self.load_skills().values()
            if (skill.skill_directory / "tool.py").is_file()
        ]

    def load_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for scope, root in self.skill_directories.items():
            for skill_path in sorted(root.glob("*/SKILL.md")):
                metadata, _ = self._load_skill_file(skill_path)
                name = str(metadata.get("name", "")).strip()
                description = str(metadata.get("description", "")).strip()
                if not name:
                    _logger.warning(f"Skill file '{skill_path}' is missing frontmatter field: name, skip it")
                    continue
                if not description:
                    _logger.warning(f"Skill file '{skill_path}' is missing frontmatter field: description, skip it")
                    continue
                if name != skill_path.parent.name:
                    _logger.warning(
                        f"Skill name {name!r} in {skill_path!r} does not match directory name "
                        f"{skill_path.parent.name!r}; use {name!r} as the skill name"
                    )
                if name in skills:
                    raise ValueError(f"Skill {name!r} is provided by more than one skill directory.")
                readonly = scope != "custom" or self._as_bool(metadata.get("readonly", False))
                skills[name] = Skill(name, description, readonly, skill_path.parent.resolve())
        return skills

    def get_skill(self, name: str) -> Skill:
        skills = self.load_skills()
        if name in skills:
            return skills[name]
        available = ", ".join(skills)
        raise ValueError(f"Skill '{name}' not found. Available skills: {available}")

    def read_skill(self, name: str, file_path: str = "SKILL.md") -> str:
        skill = self.get_skill(name)
        path = self._resolve_skill_file(skill, file_path)
        if not path.is_file():
            raise ValueError(f"Skill file not found: {file_path}")
        return path.read_text(encoding="utf-8")

    def get_skill_tree(self, name: str) -> list[str]:
        skill_dir = self.get_skill(name).skill_directory
        return [
            path.relative_to(skill_dir).as_posix() + ("/" if path.is_dir() else "")
            for path in sorted(skill_dir.rglob("*"))
            if "__pycache__" not in path.parts
            and not (path.is_file() and path.suffix in {".pyc", ".pyo"})
        ]

    def set_skill(
        self,
        name: str,
        content: str,
        file_path: str = "SKILL.md",
        force: bool = False,
    ) -> Skill:
        """Create a custom skill file or update a file in an editable skill."""
        name = self._validate_name(name)
        skills = self.load_skills()
        skill = skills.get(name)
        if skill is None:
            if file_path != "SKILL.md":
                raise ValueError("A new skill must be created by writing SKILL.md first.")
            skill_dir = (self.skill_directories["custom"] / name).resolve()
        else:
            if skill.readonly:
                raise ValueError(f"Skill '{name}' is read-only and cannot be overwritten.")
            skill_dir = skill.skill_directory

        target = self._resolve_path(skill_dir, file_path)
        if target.exists() and not force:
            raise ValueError(f"Skill file '{name}/{file_path}' already exists. Use force=True to replace it.")
        if file_path == "SKILL.md":
            metadata, _ = self._parse_skill_text(content, target)
            declared_name = str(metadata.get("name", "")).strip()
            description = str(metadata.get("description", "")).strip()
            if declared_name != name:
                raise ValueError(f"SKILL.md name {declared_name!r} must match {name!r}.")
            if not description:
                raise ValueError("SKILL.md frontmatter must contain a non-empty description.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return self.get_skill(name)

    @staticmethod
    def _validate_name(name: str) -> str:
        name = name.strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid skill name: {name!r}.")
        return name

    @classmethod
    def _resolve_skill_file(cls, skill: Skill, file_path: str) -> Path:
        return cls._resolve_path(skill.skill_directory, file_path)

    @staticmethod
    def _resolve_path(skill_dir: Path, file_path: str) -> Path:
        relative = Path(file_path or "SKILL.md")
        if relative.is_absolute():
            raise ValueError("file_path must be relative to the skill directory.")
        target = (skill_dir / relative).resolve()
        try:
            target.relative_to(skill_dir.resolve())
        except ValueError as exc:
            raise ValueError("file_path must stay inside the skill directory.") from exc
        return target

    @classmethod
    def _load_skill_file(cls, path: Path) -> tuple[dict[str, Any], str]:
        text = path.read_text(encoding="utf-8")
        metadata, normalized = cls._parse_skill_text(text, path)
        return metadata, normalized

    @staticmethod
    def _parse_skill_text(text: str, path: Path) -> tuple[dict[str, Any], str]:
        normalized = text.strip()
        if not normalized.startswith("---\n"):
            raise ValueError(f"Skill file '{path}' must start with YAML frontmatter")
        try:
            _, frontmatter, _ = normalized.split("---\n", 2)
        except ValueError as exc:
            raise ValueError(f"Skill file '{path}' has invalid YAML frontmatter") from exc
        try:
            metadata = yaml.safe_load(frontmatter.strip()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Skill file '{path}' has invalid YAML frontmatter") from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Failed to parse frontmatter in skill file '{path}', expected a YAML mapping "
                f"but got {type(metadata).__name__}"
            )
        return metadata, normalized

    @staticmethod
    def _as_bool(value: Any) -> bool:
        return str(value).lower().strip() in {"true", "yes", "1", "on"}

    @staticmethod
    def _format_skill_file(*, name: str, description: str, readonly: bool, body: str) -> str:
        metadata = yaml.safe_dump(
            {"name": name, "description": description, "readonly": readonly},
            sort_keys=False,
            allow_unicode=True,
        )
        return f"---\n{metadata}---\n\n{body}\n"
