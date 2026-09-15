# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from typing import Any, Dict, List, Tuple
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("edit_tool")
class EditTool(BaseTool):
    metadata = ToolMetadata(
        name="edit_tool",
        description=(
            "Edit and reload a previously created custom tool. Pass the skill "
            "directory name (not metadata.name) and one or more exact SEARCH/"
            "REPLACE blocks: `<<<<<<< SEARCH\\nold\\n=======\\nnew\\n>>>>>>> "
            "REPLACE`. SEARCH text must match exactly once; failed patches do "
            "not modify files. A failed reload leaves the edited tool unavailable. Custom "
            "tools must not use `@BaseTool.register(...)`; the loader registers "
            "them from `metadata.name`."
        ),
    )
    SEARCH_MARKER = "<<<<<<< SEARCH"
    SEPARATOR_MARKER = "======="
    REPLACE_MARKER = ">>>>>>> REPLACE"

    def execute(self, name: str, tool_patch: str, skill_patch: str = "") -> Dict[str, Any]:
        """Edit and reload a custom tool using exact search/replace blocks.

        Args:
            name: Exact skill name containing the custom tool.
            tool_patch: One or more exact SEARCH/REPLACE blocks for tool.py,
                using ``<<<<<<< SEARCH`` + old text + ``=======`` + new text +
                ``>>>>>>> REPLACE``. SEARCH text must match exactly once. The
                resulting tool must not use ``@BaseTool.register(...)``.
            skill_patch: Optional blocks in the same format applied to the
                complete SKILL.md file. ``name`` is the skill directory name,
                not the tool's metadata.name.
        """
        assert "skill_manager" in self.context, "skill_manager must be provided in context."
        manager = self.context["skill_manager"]
        skill = manager.get_skill(name.strip())
        if skill.readonly:
            raise ValueError(f"Skill '{name}' is read-only and cannot be edited.")
        tool_path = skill.skill_directory / "tool.py"
        old_tool = manager.read_skill(name, "tool.py")
        old_skill = manager.read_skill(name)
        new_tool, warnings = self._apply_replacements(old_tool, self._parse_patch(tool_patch))
        new_skill = old_skill
        if skill_patch.strip():
            new_skill, skill_warnings = self._apply_replacements(old_skill, self._parse_patch(skill_patch))
            warnings.extend(skill_warnings)
        if warnings:
            return {"success": False, "skill": name, "warnings": warnings, "tool_name": None}

        manager.set_skill(name, new_tool, "tool.py", force=True)
        loaded = BaseTool.load_custom_tool(tool_path)
        if new_skill != old_skill:
            manager.set_skill(name, new_skill, force=True)
        return {"success": True, "skill": name, "warnings": [], **loaded}

    def _parse_patch(self, patch: str) -> List[Tuple[str, str]]:
        lines = patch.splitlines(keepends=True)
        index, replacements = 0, []
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            if lines[index].strip() != self.SEARCH_MARKER:
                raise ValueError(f"Expected {self.SEARCH_MARKER!r} at line {index + 1}.")
            index += 1
            old_lines = []
            while index < len(lines) and lines[index].strip() != self.SEPARATOR_MARKER:
                old_lines.append(lines[index]); index += 1
            if index >= len(lines):
                raise ValueError(f"Missing {self.SEPARATOR_MARKER!r} marker in patch.")
            index += 1
            new_lines = []
            while index < len(lines) and lines[index].strip() != self.REPLACE_MARKER:
                new_lines.append(lines[index]); index += 1
            if index >= len(lines):
                raise ValueError(f"Missing {self.REPLACE_MARKER!r} marker in patch.")
            index += 1
            if not old_lines:
                raise ValueError("SEARCH block cannot be empty.")
            replacements.append(("".join(old_lines), "".join(new_lines)))
        if not replacements:
            raise ValueError("Patch must contain at least one SEARCH/REPLACE block.")
        return replacements

    def _apply_replacements(self, content: str, replacements: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
        warnings = []
        for old_text, new_text in replacements:
            count = content.count(old_text)
            if count == 0:
                warnings.append(f"SEARCH text {old_text!r} was not found.")
            elif count > 1:
                warnings.append(f"SEARCH text {old_text!r} matched more than once.")
            else:
                content = content.replace(old_text, new_text, 1)
        return content, warnings

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"Custom tool edit failed for skill {result['skill']!r}: " + "; ".join(result["warnings"])
        return f"Edited and reloaded custom tool {result['tool_name']!r} in skill {result['skill']!r}."
