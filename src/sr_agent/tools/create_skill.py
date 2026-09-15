# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import ast
import yaml
import json
import shutil
from enum import Enum, auto
from copy import deepcopy
from typing import Any, Dict
from logging import getLogger
from ..utils import bounded_value, parse_json_with_template
from .base_tool import BaseTool, ToolMetadata

_logger = getLogger(f"sr_agent.{__name__}")
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class AuthoringState(Enum):
    PREPARE = auto()
    AUTHOR = auto()
    CONTINUE = auto()
    DONE = auto()

_RESPONSE_TEMPLATE = {
    "status": str,
    "skill_type": str,
    "questions": [str],
    "draft": {
        "name": str,
        "description": str,
        "content": str,
        "tool_code": str,
        "readonly": bool,
    },
}

_DATA_ANALYSIS_DEMO = '''import numpy as np
from sr_agent.tools import BaseTool, ToolMetadata

class VariableRangeAnalysis(BaseTool):
    metadata = ToolMetadata(name="variable_range_analysis", description="Summarize one variable.")

    def execute(self, variable: str) -> dict:
        values = np.asarray(self.context["data"][variable], dtype=float)
        finite = values[np.isfinite(values)]
        return {"variable": variable, "minimum": float(finite.min()), "maximum": float(finite.max())}
'''

_FORMULA_PROPOSER_DEMO = '''from sr_agent.tools import BaseTool, ToolMetadata

class SquareFormulaProposer(BaseTool):
    metadata = ToolMetadata(name="square_formula_proposer", description="Propose a squared formula.")

    def execute(self, variable: str) -> dict:
        formula = self.parse_formula(f"{variable}**2")
        target = self.parse_formula(self.context["target"])
        return self.evaluate(f=formula, y=target)
'''

_TYPE_GUIDANCE = {
    "instructions": "Create an instruction-only skill. Leave tool_code empty and write reusable SKILL.md guidance.",
    "data_analysis": "Create a data-analysis BaseTool. Inspect self.context['data'] and return a normal analysis dictionary. Demo:\n" + _DATA_ANALYSIS_DEMO,
    "formula_proposer": "Create a formula-proposal BaseTool. Return or merge self.evaluate(...) output, preserving formula, is_candidate, and data_split_results. Demo:\n" + _FORMULA_PROPOSER_DEMO,
}

_AUTHORING_PROMPT = '''Create a reusable Agent Skill from the conversation buffer and the current create_skill request.
First decide skill_type as exactly one of instructions, data_analysis, or formula_proposer.
Use the selected type guidance and its demo. Ask only questions that cannot be resolved from context; on later internal rounds resolve them using your best supported judgment.
Do not preserve one-off answers, secrets, or unsupported guesses.
Custom tool code must import BaseTool and ToolMetadata from sr_agent.tools, define exactly one BaseTool subclass with unique metadata.name, avoid @BaseTool.register(...), and return a dict.

Return exactly one JSON object:
{"status":"needs_input" or "ready", "skill_type":"instructions" or "data_analysis" or "formula_proposer", "questions":["question"], "draft":{"name":"lowercase-hyphenated", "description":"trigger description", "content":"Markdown without frontmatter", "tool_code":"Python source or empty", "readonly":false}}
Use status=ready only when the draft is complete and questions is empty.'''


@BaseTool.register("create_skill")
class CreateSkill(BaseTool):
    metadata = ToolMetadata(name="create_skill")

    def execute(
        self,
        request: str = "",
        force: bool = False,
        history_messages: int | None = None,
    ) -> Dict[str, Any]:
        """Create a reusable skill in one tool call. The tool passes the current Agent 
        buffer and request to the configured LLM, determines whether the skill is 
        instruction-only, data-analysis, or formula-proposal, supplies the matching 
        demo, iterates internally to complete the draft, and then writes the skill.

        Args:
            request: Explain the reusable capability or lesson to capture. The current
                Agent conversation and the tool-call message are supplied automatically.
            force: Replace an existing editable skill's SKILL.md and tool.py. Other files
                are preserved; read-only skills cannot be replaced.
            history_messages: Number of most recent Agent messages to provide to the
                authoring LLM. Defaults to all available messages.
        """
        request = request.strip()
        if not request:
            raise ValueError("request must describe the skill to create.")
        if history_messages is not None:
            try:
                history_messages = int(history_messages)
            except (TypeError, ValueError) as exc:
                raise ValueError("history_messages must be a non-negative integer.") from exc
            if history_messages < 0:
                raise ValueError("history_messages must be a non-negative integer.")
        max_rounds = bounded_value(
            self.context.get("skill_authoring_rounds"),
            min=1, max=8, default=4, converter=int,
        )
        messages = []
        draft = {"name": "", "description": "", "content": "", "tool_code": "", "readonly": False}
        skill_type = ""
        questions: list[str] = []
        state = AuthoringState.PREPARE
        round_index = 0

        while True:
            if state is AuthoringState.PREPARE:
                source_messages = self.context.get("messages") or []
                if history_messages is not None:
                    source_messages = source_messages[-history_messages:] if history_messages else []
                messages.extend(deepcopy(source_messages))
                messages.append({"role": "user", "content": f"{_AUTHORING_PROMPT}\n\nCreate-skill request:\n{request}"})
                state = AuthoringState.AUTHOR

            elif state is AuthoringState.AUTHOR:
                if round_index >= max_rounds:
                    raise ValueError(f"Skill authoring did not produce a ready draft within {max_rounds} rounds.")
                guidance = _TYPE_GUIDANCE.get(skill_type, "Choose the skill type first.")
                payload = {
                    "request": request,
                    "selected_type": skill_type,
                    "guidance": guidance,
                    "current_draft": draft,
                    "pending_questions": questions,
                }
                call_messages = [*messages, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
                if callback := self.context.get("skill_authoring_callback"):
                    raw_response = callback(call_messages)
                else:
                    raw_response = self._call_configured_llm(call_messages)
                if isinstance(raw_response, dict):
                    raw_response = json.dumps(raw_response, ensure_ascii=False)
                response = parse_json_with_template(str(raw_response), _RESPONSE_TEMPLATE)
                skill_type = response["skill_type"].strip().lower()
                if skill_type not in _TYPE_GUIDANCE:
                    raise ValueError(f"Skill authoring LLM returned unsupported skill_type: {skill_type!r}")
                status = response["status"].strip().lower()
                if status not in {"needs_input", "ready"}:
                    raise ValueError(f"Skill authoring LLM returned unsupported status: {response['status']!r}")
                questions = [question.strip() for question in response["questions"] if question.strip()]
                draft = response["draft"]
                round_index += 1
                state = AuthoringState.DONE if status == "ready" else AuthoringState.CONTINUE

            elif state is AuthoringState.CONTINUE:
                messages.extend([
                    {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
                    {"role": "user", "content": (
                        "Continue internally. Resolve the pending questions from the supplied "
                        "conversation and produce the complete draft; do not wait for another user."
                    )},
                ])
                state = AuthoringState.AUTHOR

            elif state is AuthoringState.DONE:
                if questions:
                    raise ValueError("A ready skill draft must not contain clarification questions.")
                result = self._save_draft(draft, skill_type=skill_type, force=force)
                return {"status": "created", "rounds": round_index, **result}

    def _call_configured_llm(self, messages: list[dict[str, str]]) -> str:
        provider = self.context.get("llm_provider")
        model = self.context.get("llm_model")
        if not provider or not model:
            raise ValueError("create_skill requires llm_provider/llm_model context or a skill_authoring_callback.")
        from ..api import LLMAPI

        result = LLMAPI.create(provider, model)(
            messages, n=1, max_tokens=min(int(self.context.get("llm_max_tokens") or 4096), 4096),
        )
        content = ""
        for content, _, _ in result:
            pass
        if not content.strip():
            raise ValueError("Skill authoring LLM returned an empty response.")
        return content

    def _save_draft(self, draft: dict[str, Any], *, skill_type: str, force: bool) -> dict[str, Any]:
        name, description, content, tool_code, readonly = self._validate_draft(draft, skill_type)
        assert "skill_manager" in self.context, "skill_manager must be provided in context."
        manager = self.context["skill_manager"]
        existed = name in manager.load_skills()
        stored_readonly = readonly if skill_type == "instructions" else False
        skill_content = (
            "---\n"
            + yaml.safe_dump(
                {"name": name, "description": description, "readonly": stored_readonly},
                sort_keys=False,
                allow_unicode=True,
            )
            + "---\n\n"
            + content
            + "\n"
        )
        skill = manager.set_skill(name, skill_content, force=force)
        skill_dir = skill.skill_directory
        tool_path = skill_dir / "tool.py"
        if skill_type == "instructions":
            resolved = tool_path.resolve()
            for tool_name, tool_cls in list(BaseTool.REGISTRY_DICT.items()):
                if getattr(tool_cls, "source_path", None) == resolved:
                    del BaseTool.REGISTRY_DICT[tool_name]
            tool_path.unlink(missing_ok=True)
            return {"success": True, "name": name, "skill_type": skill_type, "readonly": readonly}
        manager.set_skill(name, tool_code + "\n", "tool.py", force=force)
        try:
            loaded = BaseTool.load_custom_tool(tool_path)
        except Exception:
            if not existed:
                shutil.rmtree(skill_dir, ignore_errors=True)
            raise
        if readonly:
            final_content = skill_content.replace("readonly: false", "readonly: true", 1)
            manager.set_skill(name, final_content, force=True)
        return {"success": True, "name": name, "skill_type": skill_type, "readonly": readonly, **loaded}

    @classmethod
    def _validate_draft(cls, draft: dict[str, Any], skill_type: str) -> tuple[str, str, str, str, bool]:
        name = draft["name"].strip()
        description = draft["description"].strip()
        content = draft["content"].strip()
        tool_code = draft["tool_code"].strip()
        readonly = bool(draft["readonly"])
        if not _SKILL_NAME_PATTERN.fullmatch(name):
            raise ValueError("Skill name must use lowercase letters, digits, and hyphens and cannot start or end with a hyphen.")
        if not description or not content:
            raise ValueError("Skill description and content cannot be empty.")
        if content.startswith("---"):
            try:
                _, _, content = content.split("---\n", 2)
            except ValueError as exc:
                raise ValueError("Skill content contains invalid YAML frontmatter.") from exc
            content = content.strip()
        if skill_type == "instructions" and tool_code:
            raise ValueError("Instruction-only skill drafts must leave tool_code empty.")
        if skill_type != "instructions" and not tool_code:
            raise ValueError(f"{skill_type} skill drafts must include tool_code.")
        if skill_type == "formula_proposer" and not cls._returns_base_evaluate(tool_code):
            raise ValueError("Formula-proposal tool code must return BaseTool.evaluate(...) output by calling self.evaluate(...).")
        return name, description, content, tool_code, readonly

    @classmethod
    def _returns_base_evaluate(cls, tool_code: str) -> bool:
        tree = ast.parse(tool_code)
        names = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign) and cls._is_evaluate_call(node.value) for target in node.targets if isinstance(target, ast.Name)}

        def contains(value: ast.AST | None) -> bool:
            if cls._is_evaluate_call(value):
                return True
            if isinstance(value, ast.Name):
                return value.id in names
            if isinstance(value, ast.Dict):
                return any(contains(item) for item in value.values)
            return False

        return any(isinstance(node, ast.Return) and contains(node.value) for node in ast.walk(tree))

    @staticmethod
    def _is_evaluate_call(node: ast.AST | None) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate" and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        text = f"Created {result['skill_type']} skill {result['name']!r}."
        if result.get("tool_name"):
            text += f" Registered custom tool {result['tool_name']!r}."
        return text
