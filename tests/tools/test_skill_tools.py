# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

from pathlib import Path

import pytest

from sr_agent.skills import SkillRegistry
from sr_agent.tools.create_skill import CreateSkill
from sr_agent.tools.edit_skill import EditSkill
from sr_agent.tools.read_skill import ReadSkill


class TestCreateSkillTool:
    def test_execute_creates_skill_directory_and_skill_md(self, tmp_path: Path):
        tool = CreateSkill(skills_dir=tmp_path / "skills")

        result = tool.execute(
            name="example-skill",
            description="Use this skill for examples.",
            content="# Example Skill\n\nHelpful guidance.",
        )

        skill_path = tmp_path / "skills" / "example-skill" / "SKILL.md"
        assert result["success"] is True
        assert result["name"] == "example-skill"
        assert result["path"] == str(skill_path)
        assert skill_path.exists()
        assert skill_path.read_text(encoding="utf-8").startswith("---\n")

    def test_execute_creates_parent_directory_when_missing(self, tmp_path: Path):
        skills_dir = tmp_path / "nested" / "skills"
        tool = CreateSkill(skills_dir=skills_dir)

        result = tool.execute(
            name="parent-skill",
            description="Use this skill for parent directory tests.",
            content="# Parent Skill\n\nHelpful guidance.",
        )

        assert result["success"] is True
        assert (skills_dir / "parent-skill" / "SKILL.md").exists()

    def test_execute_strips_input_frontmatter_from_content(self, tmp_path: Path):
        tool = CreateSkill(skills_dir=tmp_path / "skills")

        tool.execute(
            name="frontmatter-skill",
            description="Use this skill for frontmatter tests.",
            content="""---
name: ignored-name
description: ignored description
---

# Frontmatter Skill

Body text.
""",
        )

        skill_path = tmp_path / "skills" / "frontmatter-skill" / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")

        assert "ignored-name" not in content
        assert "# Frontmatter Skill" in content

    def test_execute_rejects_invalid_skill_name(self, tmp_path: Path):
        tool = CreateSkill(skills_dir=tmp_path / "skills")

        with pytest.raises(ValueError, match="Skill name must use lowercase letters"):
            tool.execute(
                name="Invalid Name",
                description="Use this skill for invalid name tests.",
                content="# Invalid Skill\n",
            )

    def test_execute_rejects_duplicate_skill(self, tmp_path: Path):
        tool = CreateSkill(skills_dir=tmp_path / "skills")
        tool.execute(
            name="duplicate-skill",
            description="Use this skill for duplicate tests.",
            content="# Duplicate Skill\n",
        )

        with pytest.raises(ValueError, match="already exists"):
            tool.execute(
                name="duplicate-skill",
                description="Use this skill for duplicate tests.",
                content="# Duplicate Skill\n",
            )


class TestReadSkillTool:
    def test_execute_returns_wrapped_skill_content(self, tmp_path: Path):
        create_tool = CreateSkill(skills_dir=tmp_path / "skills")
        create_tool.execute(
            name="readable-skill",
            description="Use this skill for reading tests.",
            content="# Readable Skill\n\nRead me.",
        )

        tool = ReadSkill(skills_dir=tmp_path / "skills")
        result = tool.execute(name="readable-skill")

        assert result.startswith('<skill_content name="readable-skill">')
        assert "# Readable Skill" in result
        assert "Read me." in result
        assert result.rstrip().endswith("</skill_content>")

    def test_execute_rejects_missing_skill(self, tmp_path: Path):
        tool = ReadSkill(skills_dir=tmp_path / "skills")

        with pytest.raises(ValueError, match="not found"):
            tool.execute(name="missing-skill")


class TestEditSkillTool:
    def test_execute_applies_search_replace_patch(self, tmp_path: Path):
        create_tool = CreateSkill(skills_dir=tmp_path / "skills")
        create_tool.execute(
            name="editable-skill",
            description="Use this skill for edit tests.",
            content="# Editable Skill\n\nOld text.\n",
        )

        tool = EditSkill(skills_dir=tmp_path / "skills")
        result = tool.execute(
            name="editable-skill",
            patch=(
                "<<<<<<< SEARCH\n"
                "Old text.\n"
                "=======\n"
                "New text.\n"
                ">>>>>>> REPLACE\n"
            ),
        )

        skill_path = tmp_path / "skills" / "editable-skill" / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")

        assert result["message"] == "Edited skill 'editable-skill' with 1 replacement(s)."
        assert "exceptions" in result
        assert "New text." in content
        assert "Old text." not in content

    def test_execute_rejects_missing_skill(self, tmp_path: Path):
        tool = EditSkill(skills_dir=tmp_path / "skills")

        with pytest.raises(ValueError, match="does not exist"):
            tool.execute(
                name="missing-skill",
                patch=(
                    "<<<<<<< SEARCH\n"
                    "Old text.\n"
                    "=======\n"
                    "New text.\n"
                    ">>>>>>> REPLACE\n"
                ),
            )

    def test_execute_rejects_read_only_skill(self, tmp_path: Path):
        create_tool = CreateSkill(skills_dir=tmp_path / "skills")
        create_tool.execute(
            name="readonly-skill",
            description="Use this skill for readonly tests.",
            content="# Readonly Skill\n\nOld text.\n",
            readonly=True,
        )

        tool = EditSkill(skills_dir=tmp_path / "skills")
        with pytest.raises(ValueError, match="read-only"):
            tool.execute(
                name="readonly-skill",
                patch=(
                    "<<<<<<< SEARCH\n"
                    "Old text.\n"
                    "=======\n"
                    "New text.\n"
                    ">>>>>>> REPLACE\n"
                ),
            )

    def test_execute_rejects_patch_with_ambiguous_search_text(self, tmp_path: Path):
        create_tool = CreateSkill(skills_dir=tmp_path / "skills")
        create_tool.execute(
            name="ambiguous-skill",
            description="Use this skill for ambiguous edit tests.",
            content="# Ambiguous Skill\n\nSame.\nSame.\n",
        )

        tool = EditSkill(skills_dir=tmp_path / "skills")
        result = tool.execute(
            name="ambiguous-skill",
            patch=(
                "<<<<<<< SEARCH\n"
                "Same.\n"
                "=======\n"
                "Different.\n"
                ">>>>>>> REPLACE\n"
            ),
        )

        assert result["exceptions"]
        assert "matched more than once" in result["exceptions"][0]

    def test_execute_rejects_missing_search_marker(self, tmp_path: Path):
        create_tool = CreateSkill(skills_dir=tmp_path / "skills")
        create_tool.execute(
            name="marker-skill",
            description="Use this skill for marker tests.",
            content="# Marker Skill\n\nOld text.\n",
        )

        tool = EditSkill(skills_dir=tmp_path / "skills")
        with pytest.raises(ValueError, match="Expected '<<<<<<< SEARCH'"):
            tool.execute(
                name="marker-skill",
                patch=(
                    "Old text.\n"
                    "=======\n"
                    "New text.\n"
                    ">>>>>>> REPLACE\n"
                ),
            )
