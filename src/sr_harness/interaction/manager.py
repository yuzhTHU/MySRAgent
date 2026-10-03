"""Frontend boundary used by :class:`SRAgentInteractive`."""
from __future__ import annotations

from abc import ABC
from typing import Any


class InteractionManager(ABC):
    """Connect an interactive agent to a user interface.

    The default implementation is intentionally inert. Frontends may override
    control, prompt preparation, workspace ownership, and event publication
    without taking ownership of the search loop.
    """

    def bind_run_state(self, run_state) -> None:
        """Expose the authoritative in-memory run state to the frontend."""

    def bind_workspace(self, workspace) -> None:
        """Expose the active workspace to the frontend."""

    def prepare_initial_prompt(self, messages, *, X, y):
        """Apply frontend-owned prompt additions or user overrides."""
        return messages

    def checkpoint(self) -> list[str]:
        """Wait at a safe boundary and return queued human guidance."""
        return []

    def wait_until_running(self) -> None:
        """Wait at a tool boundary while the frontend has paused the run."""

    def take_model_settings(self) -> dict[str, Any] | None:
        """Return and consume model settings queued by the frontend."""
        return None

    def commit_model_settings(self, settings: dict[str, Any]) -> None:
        """Tell the frontend that queued model settings were applied."""

    def ask_human(self, message: str) -> str:
        """Ask the connected user for guidance."""
        raise RuntimeError("This interaction manager cannot ask a user for input.")

    def publish(self, kind: str, payload: Any) -> None:
        """Publish an observable event to the frontend."""
