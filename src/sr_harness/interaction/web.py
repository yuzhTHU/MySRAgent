# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Web adapter for an interactive SRHarness run."""
from __future__ import annotations

import shutil

from ..core import json_value
from .manager import InteractionManager


def add_variable_descriptions(messages, descriptions, variables):
    rows = [
        f"- {name}: {descriptions[name]}"
        for name in variables
        if descriptions.get(name)
    ]
    if rows:
        for message in messages:
            if message.get("role") == "user":
                message["content"] += "\n\nVariable descriptions:\n" + "\n".join(rows)
                break
    return messages


class WebInteractionManager(InteractionManager):
    """Connect one interactive agent to an :class:`InteractiveSession`."""

    def __init__(self, session):
        self.session = session

    def bind_run_state(self, run_state) -> None:
        with self.session.lock:
            self.session.run_state = run_state

    def bind_workspace(self, workspace) -> None:
        workspace.retain = True
        with self.session.lock:
            if self.session.workspace != workspace.path:
                for item in self.session.workspace.iterdir():
                    shutil.move(str(item), str(workspace.path / item.name))
                self.session.workspace = workspace.path

    def prepare_initial_prompt(self, messages, *, X, y):
        with self.session.lock:
            overrides = self.session.prompt_overrides.copy()
            descriptions = self.session.variable_descriptions.copy()
        add_variable_descriptions(messages, descriptions, [*X, *y])
        for message in messages:
            if message.get("role") in overrides:
                message["content"] = overrides[message["role"]]
        return messages

    def checkpoint(self) -> list[str]:
        return self.session.controller.checkpoint()

    def wait_until_running(self) -> None:
        self.session.controller.wait_until_running()

    def take_model_settings(self):
        with self.session.lock:
            settings = self.session.pending_settings
            self.session.pending_settings = None
        return settings

    def commit_model_settings(self, settings) -> None:
        with self.session.lock:
            self.session.settings.update(settings)

    def ask_human(self, message: str) -> str:
        return self.session.controller.ask(message)

    def publish(self, kind: str, payload) -> None:
        self.session.controller.publish(kind, json_value(payload))
