# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sr_agent.skills import SkillManager
from sr_agent.tools.create_skill import CreateSkill
from sr_agent.tools.edit_skill import EditSkill
from sr_agent.tools.edit_tool import EditTool
from sr_agent.tools.read_skill import ReadSkill
from sr_agent.tools.base_tool import BaseTool
from sr_agent.tools.workspace_shell import WorkspaceShellTool


def _test_manager(custom_directory: Path) -> SkillManager:
    return SkillManager(
        built_in_directory=custom_directory.parent / "empty-built-in-skills",
        custom_directory=custom_directory,
    )


def _discover_and_load_tools(custom_directory: Path) -> list[dict]:
    classes = BaseTool.discover_custom_tools(_test_manager(custom_directory))
    return [
        {
            "tool_name": tool_cls.metadata.name,
            "path": str(tool_cls.source_path),
            "class_name": tool_cls.__name__,
        }
        for tool_cls in classes
    ]


def _write_skill(skills_dir: Path, name: str, content: str, *, readonly: bool = False) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=False)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Use this skill for {name} tests.\n"
        f"readonly: {'true' if readonly else 'false'}\n"
        "---\n\n"
        f"{content.strip()}\n",
        encoding="utf-8",
    )
    return skill_path


def _draft(
    name: str = "example-skill",
    content: str = "# Example Skill\n\nHelpful guidance.",
    tool_code: str = "",
    readonly: bool = False,
) -> dict:
    return {
        "status": "ready",
        "skill_type": "instructions" if not tool_code else "data_analysis",
        "questions": [],
        "draft": {
            "name": name,
            "description": "Use this skill for reusable examples.",
            "content": content,
            "tool_code": tool_code,
            "readonly": readonly,
        },
    }


def _create_authored_skill(
    skills_dir: Path,
    *,
    skill_type: str = "instructions",
    force: bool = False,
    **draft_kwargs,
) -> dict:
    draft = _draft(**draft_kwargs)
    draft["skill_type"] = skill_type
    tool = CreateSkill(
        skill_manager=_test_manager(skills_dir),
        skill_authoring_callback=lambda messages: draft,
    )
    return tool.execute(request="Create a reusable test skill.", force=force)


class _SkillCreatorHarness:
    """Exercise the unified CreateSkill flow with a deterministic authored draft."""

    def __init__(self, *, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skill_manager = _test_manager(skills_dir)

    def execute(
        self,
        *,
        tool_type: str,
        name: str,
        description: str,
        content: str,
        tool_code: str,
        readonly: bool = False,
        force: bool = False,
    ) -> dict:
        response = _draft(
            name=name,
            content=content,
            tool_code=tool_code,
            readonly=readonly,
        )
        response["draft"]["description"] = description
        response["skill_type"] = tool_type
        creator = CreateSkill(
            skill_manager=self.skill_manager,
            skill_authoring_callback=lambda messages: response,
        )
        return creator.execute(request="Create a custom tool skill.", force=force)


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Snapshot and restore the global BaseTool registry around each test."""
    snapshot = dict(BaseTool.REGISTRY_DICT)
    yield
    BaseTool.REGISTRY_DICT.clear()
    BaseTool.REGISTRY_DICT.update(snapshot)


class TestSkillManager:
    def test_directories_are_paths_and_tool_docs_are_runtime_skills(self, tmp_path):
        registry = _test_manager(tmp_path / "custom")
        assert all(isinstance(path, Path) for path in registry.skill_directories.values())
        assert isinstance(WorkspaceShellTool.get_doc(), dict)

        registry.register_tool_docs([WorkspaceShellTool])

        skill = registry.get_skill("workspace-shell")
        assert skill.readonly is True
        assert skill.skill_directory.parent == registry.skill_directories["runtime"]
        assert "# Workspace Shell" in registry.read_skill("workspace-shell")

    def test_discovers_tool_skills_across_all_managed_directories(self, tmp_path):
        built_in = tmp_path / "built-in"
        custom = tmp_path / "custom"
        _write_skill(built_in, "built-in-tool", "# Built-in")
        _write_skill(custom, "custom-tool", "# Custom")
        manager = SkillManager(built_in_directory=built_in, custom_directory=custom)
        manager.register_tool_docs([WorkspaceShellTool])
        for name in ("built-in-tool", "custom-tool", "workspace-shell"):
            (manager.get_skill(name).skill_directory / "tool.py").write_text("# tool\n")

        assert {skill.name for skill in manager.discover_tool_skills()} == {
            "built-in-tool", "custom-tool", "workspace-shell",
        }

    def test_set_skill_requires_force_and_never_overwrites_readonly(self, tmp_path):
        registry = _test_manager(tmp_path / "custom")
        original = "---\nname: editable\ndescription: old\nreadonly: false\n---\n\nOld\n"
        replacement = original.replace("old", "new").replace("Old", "New")
        registry.set_skill("editable", original)

        with pytest.raises(ValueError, match="force=True"):
            registry.set_skill("editable", replacement)
        registry.set_skill("editable", replacement, force=True)
        assert "New" in registry.read_skill("editable")

        readonly = replacement.replace("readonly: false", "readonly: true")
        registry.set_skill("editable", readonly, force=True)
        with pytest.raises(ValueError, match="read-only"):
            registry.set_skill("editable", original, force=True)


class TestCreateSkillTool:
    TOOL_CODE = '''
from sr_agent.tools import BaseTool, ToolMetadata
class ExampleAnalysis(BaseTool):
    metadata = ToolMetadata(name="example_analysis", description="Analyze a value.")
    def execute(self, value: float = 0.0):
        """Analyze a value."""
        return {"value": value}
'''
    FORMULA_TOOL_CODE = '''
from sr_agent.tools import BaseTool, ToolMetadata
class ExampleProposer(BaseTool):
    metadata = ToolMetadata(name="example_proposer", description="Propose a formula.")
    def execute(self, variable: str):
        """Propose a formula."""
        return self.evaluate(f=self.parse_formula(variable), y=self.parse_formula(self.context["target"]))
'''

    def test_execute_creates_instruction_skill(self, tmp_path: Path):
        result = _create_authored_skill(tmp_path / "skills")
        assert result["status"] == "created"
        assert (tmp_path / "skills" / "example-skill" / "SKILL.md").exists()

    def test_execute_creates_formula_tool_and_registers_it(self, tmp_path: Path):
        result = _create_authored_skill(
            tmp_path / "skills", skill_type="formula_proposer",
            name="example-formula-proposer", tool_code=self.FORMULA_TOOL_CODE,
        )
        assert result["tool_name"] == "example_proposer"

    def test_execute_passes_buffer_to_authoring_callback(self, tmp_path: Path):
        captured = {}
        def author(messages):
            captured["messages"] = messages
            return _draft()
        tool = CreateSkill(
            skill_manager=_test_manager(tmp_path / "skills"), skill_authoring_callback=author,
            messages=[
                {"role": "system", "content": "System context"},
                {"role": "assistant", "content": "I will call create_skill now."},
            ],
        )
        tool.execute(request="Capture this reusable workflow.")
        serialized = json.dumps(captured["messages"], ensure_ascii=False)
        assert "System context" in serialized
        assert "I will call create_skill now." in serialized
        assert "Capture this reusable workflow." in serialized


class TestReadSkillTool:
    def test_execute_returns_wrapped_skill_content(self, tmp_path: Path):
        _write_skill(tmp_path / "skills", "readable-skill", "# Readable Skill\n\nRead me.")

        tool = ReadSkill(skill_manager=_test_manager(tmp_path / "skills"))
        result = tool.execute(name="readable-skill")

        assert result["content"].startswith('<skill_content name="readable-skill">')
        assert "# Readable Skill" in result["content"]
        assert "Read me." in result["content"]
        assert result["content"].rstrip().endswith("</skill_content>")

    def test_call_returns_dict_result_and_wrapped_content_string(self, tmp_path: Path):
        _write_skill(
            tmp_path / "skills",
            "callable-skill",
            "# Callable Skill\n\nRead me through BaseTool.",
        )

        result = ReadSkill(skill_manager=_test_manager(tmp_path / "skills"))(name="callable-skill")

        assert result.ok is True
        assert set(result.result) == {"content"}
        assert result.get("metrics") is None
        assert result.result_str.startswith('<skill_content name="callable-skill">')
        assert "# Callable Skill" in result.result_str

    def test_execute_rejects_missing_skill(self, tmp_path: Path):
        tool = ReadSkill(skill_manager=_test_manager(tmp_path / "skills"))

        with pytest.raises(ValueError, match="not found"):
            tool.execute(name="missing-skill")


class TestEditSkillTool:
    def test_execute_applies_search_replace_patch(self, tmp_path: Path):
        _write_skill(tmp_path / "skills", "editable-skill", "# Editable Skill\n\nOld text.\n")

        tool = EditSkill(skill_manager=_test_manager(tmp_path / "skills"))
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

        assert result["skill"] == "editable-skill"
        assert result["applied_replacements"] == 1
        assert result["exceptions"] == []
        assert "New text." in content
        assert "Old text." not in content

    def test_execute_rejects_missing_skill(self, tmp_path: Path):
        tool = EditSkill(skill_manager=_test_manager(tmp_path / "skills"))

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
        _write_skill(
            tmp_path / "skills",
            "readonly-skill",
            "# Readonly Skill\n\nOld text.\n",
            readonly=True,
        )

        tool = EditSkill(skill_manager=_test_manager(tmp_path / "skills"))
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
        _write_skill(
            tmp_path / "skills",
            "ambiguous-skill",
            "# Ambiguous Skill\n\nSame.\nSame.\n",
        )

        tool = EditSkill(skill_manager=_test_manager(tmp_path / "skills"))
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

        assert result["applied_replacements"] == 0
        assert "matched more than once" in result["exceptions"][0]

    def test_execute_rejects_missing_search_marker(self, tmp_path: Path):
        _write_skill(tmp_path / "skills", "marker-skill", "# Marker Skill\n\nOld text.\n")

        tool = EditSkill(skill_manager=_test_manager(tmp_path / "skills"))
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


class TestCreateSkillWithTool:
    TOOL_CODE = '''
from sr_agent.tools import BaseTool, ToolMetadata

class DoubleValue(BaseTool):
    metadata = ToolMetadata(name="double_value", description="Doubles the input value.")
    def execute(self, value: float = 0.0):
        """Double the input.

        Args:
            value: the number to double.
        """
        return {"value": value * 2}
'''

    def test_formula_proposer_requires_base_evaluate(self, tmp_path):
        with pytest.raises(ValueError, match="must return BaseTool.evaluate"):
            _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
                tool_type="formula_proposer",
                name="invalid-proposer",
                description="Propose an invalid formula.",
                content="# Invalid proposer",
                tool_code=self.TOOL_CODE,
            )

    def test_execute_creates_tool_and_registers(self, tmp_path):
        from sr_agent.tools import BaseTool
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        result = tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Double Skill\\n\\nDoubles numbers.",
            tool_code=self.TOOL_CODE,
        )
        assert result["success"] is True
        assert result["tool_name"] == "double_value"
        tool_path = tmp_path / "skills" / "double-value" / "tool.py"
        assert tool_path.exists()
        assert "double_value" in BaseTool.REGISTRY_DICT
        assert BaseTool.REGISTRY_DICT["double_value"].source_path == tool_path.resolve()

    def test_execute_registered_tool_can_be_called(self, tmp_path):
        from sr_agent.tools import BaseTool
        _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Double Skill\\n\\nDoubles numbers.",
            tool_code=self.TOOL_CODE,
        )
        ready = BaseTool.create("double_value", skills_dir=tmp_path / "skills")
        assert ready.execute(value=21)["value"] == 42

    def test_execute_rejects_duplicate_tool_name(self, tmp_path):
        from sr_agent.tools import BaseTool
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Double Skill",
            tool_code=self.TOOL_CODE,
        )
        with pytest.raises(ValueError, match="already registered"):
            tool.execute(
                tool_type="data_analysis",
                name="other-skill",
                description="Use to double again.",
                content="# Other Skill",
                tool_code=self.TOOL_CODE,
            )
        # failed creation must not leave the skill directory behind
        assert not (tmp_path / "skills" / "other-skill").exists()
        assert "double_value" in BaseTool.REGISTRY_DICT

    def test_execute_rolls_back_on_bad_syntax(self, tmp_path):
        from sr_agent.tools import BaseTool
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        bad = "def execute(self):\\n    return {  # unterminated\\n"
        for name in ("bad-skill",):
            before = set(BaseTool.REGISTRY_DICT)
            with pytest.raises(SyntaxError):
                tool.execute(
                    tool_type="data_analysis",
                    name=name,
                    description="bad",
                    content="# Bad",
                    tool_code=bad,
                )
            assert not (tmp_path / "skills" / name).exists()
            assert set(BaseTool.REGISTRY_DICT) == before

    def test_execute_rejects_registration_decorator(self, tmp_path):
        decorated_code = self.TOOL_CODE.replace(
            "class DoubleValue(BaseTool):",
            '@BaseTool.register("double_value")\nclass DoubleValue(BaseTool):',
        )

        with pytest.raises(ValueError, match="must not use @BaseTool.register"):
            _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
                tool_type="data_analysis",
                name="decorated-tool",
                description="Must be rejected.",
                content="# Decorated Tool",
                tool_code=decorated_code,
            )

        assert not (tmp_path / "skills" / "decorated-tool").exists()

    def test_execute_force_replaces_tool_files_but_preserves_other_files(self, tmp_path):
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Old Skill",
            tool_code=self.TOOL_CODE,
        )
        notes_path = tmp_path / "skills" / "double-value" / "notes.md"
        notes_path.write_text("keep me", encoding="utf-8")
        updated_code = self.TOOL_CODE.replace("value * 2", "value * 4")

        tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to quadruple a value.",
            content="# New Skill",
            tool_code=updated_code,
            force=True,
        )

        assert notes_path.read_text(encoding="utf-8") == "keep me"
        assert "# New Skill" in (notes_path.parent / "SKILL.md").read_text(encoding="utf-8")
        assert BaseTool.create("double_value").execute(value=3)["value"] == 12

    def test_execute_force_cannot_replace_readonly_skill(self, tmp_path):
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Read-only Skill",
            tool_code=self.TOOL_CODE,
            readonly=True,
        )

        with pytest.raises(ValueError, match="read-only"):
            tool.execute(
                tool_type="data_analysis",
                name="double-value",
                description="Replace it.",
                content="# Replacement",
                tool_code=self.TOOL_CODE,
                force=True,
            )

    def test_failed_force_replace_keeps_new_files_and_disables_old_tool(self, tmp_path):
        tool = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        tool.execute(
            tool_type="data_analysis",
            name="double-value",
            description="Use to double a value.",
            content="# Working Skill",
            tool_code=self.TOOL_CODE,
        )

        with pytest.raises(SyntaxError):
            tool.execute(
                tool_type="data_analysis",
                name="double-value",
                description="Broken replacement.",
                content="# Broken Skill",
                tool_code="broken python {",
                force=True,
            )

        skill_dir = tmp_path / "skills" / "double-value"
        assert (skill_dir / "tool.py").read_text(encoding="utf-8") == "broken python {\n"
        assert "# Broken Skill" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "double_value" not in BaseTool.REGISTRY_DICT


class TestEditToolTool:
    TOOL_CODE = '''
from sr_agent.tools import BaseTool, ToolMetadata

class TripleValue(BaseTool):
    metadata = ToolMetadata(name="triple_value", description="Triples the input.")
    def execute(self, value: float = 0.0):
        """Triple the input.

        Args:
            value: the number to triple.
        """
        return {"value": value * 3}
'''

    def test_execute_edits_and_reloads_tool(self, tmp_path):
        from sr_agent.tools import BaseTool
        create = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        create.execute(
            tool_type="data_analysis",
            name="triple-skill",
            description="Use to triple a value.",
            content="# Triple Skill",
            tool_code=self.TOOL_CODE,
        )
        edit = EditTool(skill_manager=_test_manager(tmp_path / "skills"))
        patch = (
            "<<<<<<< SEARCH\n"
            "value * 3}\n"
            "=======\n"
            "value * 4}\n"
            ">>>>>>> REPLACE\n"
        )
        result = edit.execute(name="triple-skill", tool_patch=patch)
        assert result["success"] is True
        ready = BaseTool.create("triple_value", skills_dir=tmp_path / "skills")
        assert ready.execute(value=21)["value"] == 84

    def test_execute_rejects_ambiguous_search(self, tmp_path):
        tool_code = "# placeholder\n# placeholder\n"
        create = _SkillCreatorHarness(skills_dir=tmp_path / "skills")
        create.execute(
            tool_type="data_analysis",
            name="ambig-skill",
            description="Use to triple a value.",
            content="# Ambig Skill",
            tool_code="\n".join(
                [
                    "from sr_agent.tools import BaseTool, ToolMetadata",
                    "class AmbigTool(BaseTool):",
                    "    metadata = ToolMetadata(name=\"ambig_tool\", description=\"d\")",
                    "    def execute(self):",
                    "        \"\"\"doc\"\"\"",
                    "        # placeholder",
                    "        # placeholder",
                    "        return {}",
                    "",
                ]
            ),
        )
        edit = EditTool(skill_manager=_test_manager(tmp_path / "skills"))
        patch = (
            "<<<<<<< SEARCH\n"
            "# placeholder\n"
            "=======\n"
            "# replaced\n"
            ">>>>>>> REPLACE\n"
        )
        result = edit.execute(name="ambig-skill", tool_patch=patch)
        assert result["success"] is False
        assert any("more than once" in w for w in result["warnings"])
        # files must be untouched on ambiguous match
        content = (tmp_path / "skills" / "ambig-skill" / "tool.py").read_text(encoding="utf-8")
        assert "# replaced" not in content
        assert content.count("# placeholder") == 2

    def test_failed_reload_keeps_edit_and_disables_tool(self, tmp_path):
        _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
            tool_type="data_analysis",
            name="triple-skill",
            description="Use to triple a value.",
            content="# Triple Skill",
            tool_code=self.TOOL_CODE,
        )
        tool_path = tmp_path / "skills" / "triple-skill" / "tool.py"

        with pytest.raises(ValueError, match="BaseTool subclass"):
            EditTool(skill_manager=_test_manager(tmp_path / "skills")).execute(
                name="triple-skill",
                tool_patch=(
                    "<<<<<<< SEARCH\n"
                    "class TripleValue(BaseTool):\n"
                    "=======\n"
                    "class TripleValue(object):\n"
                    ">>>>>>> REPLACE\n"
                ),
            )

        assert "class TripleValue(object):" in tool_path.read_text(encoding="utf-8")
        assert "triple_value" not in BaseTool.REGISTRY_DICT


class TestReadSkillFile:
    def test_execute_reads_tool_file_and_tree(self, tmp_path):
        from sr_agent.tools import BaseTool
        code = '''
from sr_agent.tools import BaseTool, ToolMetadata
class FileTool(BaseTool):
    metadata = ToolMetadata(name="file_tool", description="d")
    def execute(self):
        """doc"""
        return {}
'''
        _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
            tool_type="data_analysis",
            name="file-skill",
            description="d",
            content="# File Skill",
            tool_code=code,
        )
        read = ReadSkill(skill_manager=_test_manager(tmp_path / "skills"))
        result = read.execute(name="file-skill", file_path="tool.py", show_tree=True)
        assert result["file_path"] == "tool.py"
        assert "FileTool" in result["file_content"]
        assert "tool.py" in result["tree"]
        # __pycache__ must be excluded from the tree
        assert not any("__pycache__" in p for p in result["tree"])

    def test_execute_rejects_path_traversal(self, tmp_path):
        from sr_agent.tools import BaseTool
        code = '''
from sr_agent.tools import BaseTool, ToolMetadata
class TravTool(BaseTool):
    metadata = ToolMetadata(name="trav_tool", description="d")
    def execute(self):
        """doc"""
        return {}
'''
        _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
            tool_type="data_analysis",
            name="trav-skill",
            description="d",
            content="# Trav Skill",
            tool_code=code,
        )
        read = ReadSkill(skill_manager=_test_manager(tmp_path / "skills"))
        with pytest.raises(ValueError, match="stay inside"):
            read.execute(name="trav-skill", file_path="../other.md")


class TestDiscoverToolSkills:
    """Tool-bearing skills are discovered and loaded independently."""

    def _write_saved_tool(self, tmp_path, name, reg_name, expr):
        """Write a saved custom tool file directly, as a prior session would."""
        code = (
            f"from sr_agent.tools import BaseTool, ToolMetadata\n"
            f"class Tool(BaseTool):\n"
            f"    metadata = ToolMetadata(name=\"{reg_name}\", description=\"d\")\n"
            f"    def execute(self, x: float):\n"
            f"        \"\"\"doc\"\"\"\n"
            f"        return {{\"result\": {expr}}}\n"
        )
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\n# S\n", encoding="utf-8")
        (skill_dir / "tool.py").write_text(code, encoding="utf-8")

    def test_discover_registers_saved_tools(self, tmp_path):
        self._write_saved_tool(tmp_path, "square-skill", "square_it", "x * x")
        loaded = _discover_and_load_tools(tmp_path / "skills")
        assert len(loaded) == 1
        assert loaded[0]["tool_name"] == "square_it"
        assert "square_it" in BaseTool.REGISTRY_DICT

    def test_repeated_agents_do_not_duplicate_discovered_tool_classes(self, tmp_path, monkeypatch):
        import importlib

        sr_agent_module = importlib.import_module("sr_agent.sr_agent")
        custom_directory = tmp_path / "skills"
        self._write_saved_tool(tmp_path, "square-skill", "square_it", "x * x")
        monkeypatch.setattr(
            sr_agent_module,
            "SkillManager",
            lambda: _test_manager(custom_directory),
        )

        agents = [
            sr_agent_module.SRAgent(
                llm_provider="unused",
                llm_model="unused",
                tools=["read_skill", "square_it"],
                save_path=str(tmp_path / f"agent-{index}"),
            )
            for index in range(2)
        ]

        for agent in agents:
            names = [tool_cls.metadata.name for tool_cls in agent.tool_cls_list]
            assert names.count("square_it") == 1

    def test_discover_reloads_registered_tools(self, tmp_path):
        self._write_saved_tool(tmp_path, "cube-skill", "cube_it", "x ** 3")
        _discover_and_load_tools(tmp_path / "skills")
        second = _discover_and_load_tools(tmp_path / "skills")
        assert [item["tool_name"] for item in second] == ["cube_it"]
        assert "cube_it" in BaseTool.REGISTRY_DICT

    def test_discover_reloads_modified_tool(self, tmp_path):
        self._write_saved_tool(tmp_path, "power-skill", "power_it", "x ** 2")
        _discover_and_load_tools(tmp_path / "skills")
        self._write_saved_tool(tmp_path, "power-skill", "power_it", "x ** 4")

        loaded = _discover_and_load_tools(tmp_path / "skills")

        assert [item["tool_name"] for item in loaded] == ["power_it"]
        assert BaseTool.create("power_it").execute(x=3)["result"] == 81

    def test_discover_replaces_old_name_when_metadata_name_changes(self, tmp_path):
        self._write_saved_tool(tmp_path, "rename-skill", "old_name", "x + 1")
        _discover_and_load_tools(tmp_path / "skills")
        self._write_saved_tool(tmp_path, "rename-skill", "new_name", "x + 1")

        _discover_and_load_tools(tmp_path / "skills")

        assert "old_name" not in BaseTool.REGISTRY_DICT
        assert "new_name" in BaseTool.REGISTRY_DICT

    def test_discover_broken_update_disables_old_tool(self, tmp_path):
        self._write_saved_tool(tmp_path, "fragile-skill", "fragile_tool", "x + 1")
        _discover_and_load_tools(tmp_path / "skills")
        tool_path = tmp_path / "skills" / "fragile-skill" / "tool.py"
        tool_path.write_text("broken python {", encoding="utf-8")

        loaded = _discover_and_load_tools(tmp_path / "skills")

        assert loaded == []
        assert "fragile_tool" not in BaseTool.REGISTRY_DICT

    def test_discover_skips_second_file_with_duplicate_name(self, tmp_path):
        self._write_saved_tool(tmp_path, "a-skill", "shared_tool", "x + 1")
        self._write_saved_tool(tmp_path, "b-skill", "shared_tool", "x + 2")

        loaded = _discover_and_load_tools(tmp_path / "skills")

        assert [item["tool_name"] for item in loaded] == ["shared_tool"]
        assert BaseTool.create("shared_tool").execute(x=1)["result"] == 2

    def test_discover_does_not_unregister_deleted_tool_file(self, tmp_path):
        self._write_saved_tool(tmp_path, "deleted-skill", "deleted_tool", "x + 1")
        _discover_and_load_tools(tmp_path / "skills")
        (tmp_path / "skills" / "deleted-skill" / "tool.py").unlink()

        loaded = _discover_and_load_tools(tmp_path / "skills")

        assert loaded == []
        assert "deleted_tool" in BaseTool.REGISTRY_DICT

    def test_discover_skips_broken_tool_without_blocking(self, tmp_path):
        skills = tmp_path / "skills"
        self._write_saved_tool(tmp_path, "ok-skill", "ok_tool", "x + 1")
        bad = skills / "bad-skill"
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "tool.py").write_text("def this_is_not_a_valid_tool():\n    pass\n", encoding="utf-8")
        loaded = _discover_and_load_tools(skills)
        assert [d["tool_name"] for d in loaded] == ["ok_tool"]
        assert "ok_tool" in BaseTool.REGISTRY_DICT

    def test_create_skill_registers_tool_immediately(self, tmp_path):
        code = (
            "from sr_agent.tools import BaseTool, ToolMetadata\n"
            "class Tool(BaseTool):\n"
            "    metadata = ToolMetadata(name=\"create_discovered\", description=\"d\")\n"
            "    def execute(self):\n"
            "        \"\"\"doc\"\"\"\n"
            "        return {}\n"
        )
        _SkillCreatorHarness(skills_dir=tmp_path / "skills").execute(
            tool_type="data_analysis", name="auto-skill", description="d", content="# S", tool_code=code,
        )
        # create_skill registers the new tool in the current process immediately
        assert "create_discovered" in BaseTool.REGISTRY_DICT
