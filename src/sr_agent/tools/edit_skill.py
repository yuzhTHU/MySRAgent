# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from ..skills import SkillRegistry
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("edit_skill")
class EditSkill(BaseTool):
    metadata = ToolMetadata(name="edit_skill")
    SEARCH_MARKER = "<<<<<<< SEARCH"
    SEPARATOR_MARKER = "======="
    REPLACE_MARKER = ">>>>>>> REPLACE"

    def execute(self, name: str, patch: str) -> Dict[str, Any]:
        """Edit an existing skill with exact search/replace blocks.

        Args:
            name: The exact skill name to edit.
            patch: One or more SEARCH/REPLACE blocks. Each SEARCH text must 
                match exactly once in the target skill file. Format:

                <<<<<<< SEARCH
                old text copied exactly from the skill file
                =======
                new replacement text
                >>>>>>> REPLACE

                Use multiple blocks for multiple focused edits.
        """
        name = name.strip()
        replacements = self._parse_patch(patch)

        skills_dir = self.context.get("skills_dir") # 仅供调试使用，一般用不到
        registry = SkillRegistry(skills_dir)
        if name not in registry.load_skills():
            raise ValueError(f"Skill '{name}' does not exist and cannot be edited.")
        elif (skill := registry.get_skills(name)).readonly:
            raise ValueError(f"Skill '{name}' is read-only and cannot be edited.")
        else:
            old_content = skill.path.read_text(encoding="utf-8")
            new_content, exceptions = self._apply_replacements(old_content, replacements)
            skill.path.write_text(new_content, encoding="utf-8")
            return {
                "message": f"Edited skill {name!r} with {len(replacements)} replacement(s).",
                "exceptions": exceptions,
            }

    def _parse_patch(self, patch: str) -> List[Tuple[str, str]]:
        lines = patch.splitlines(keepends=True)

        index = 0
        replacements = []
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            if lines[index].strip() != self.SEARCH_MARKER:
                raise ValueError(
                    f"Expected {self.SEARCH_MARKER!r} at line {index + 1}, "
                    f"got {lines[index].strip()!r}."
                )

            index += 1
            old_lines = []
            while index < len(lines) and lines[index].strip() != self.SEPARATOR_MARKER:
                old_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ValueError(f"Missing {self.SEPARATOR_MARKER!r} marker in patch.")

            index += 1
            new_lines = []
            while index < len(lines) and lines[index].strip() != self.REPLACE_MARKER:
                new_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ValueError(f"Missing {self.REPLACE_MARKER!r} marker in patch.")

            index += 1
            old_text = "".join(old_lines)
            new_text = "".join(new_lines)
            if not old_text:
                raise ValueError("SEARCH block cannot be empty.")
            replacements.append((old_text, new_text))

        if not replacements:
            raise ValueError("Patch must contain at least one SEARCH/REPLACE block.")

        return replacements

    def _apply_replacements(self, content: str, replacements: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
        exceptions = []
        for old_text, new_text in replacements:
            if (count := content.count(old_text)) == 0:
                exceptions.append(
                    f"SEARCH text {old_text!r} was not found in "
                    f"the skill file, skip this replacement."
                )
            elif count > 1:
                exceptions.append(
                    f"SEARCH text {old_text!r} matched more than "
                    f"once. Include more surrounding context."
                )
            else:
                content = content.replace(old_text, new_text, 1)
        return content, exceptions
