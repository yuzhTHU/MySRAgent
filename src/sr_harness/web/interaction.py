# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Thread-safe bidirectional control shared by interactive and web frontends."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any


class InteractionController:
    """Coordinate pause/resume/stop, injected guidance, events, and replies."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._paused = False
        self._stopped = False
        self._events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._guidance: deque[str] = deque()
        self._replies: dict[str, str] = {}
        self._sequence = 0

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "paused": self._paused,
                "stopped": self._stopped,
                "pending_guidance": len(self._guidance),
                "last_event_seq": self._sequence,
            }

    def command(self, action: str, message: str = "") -> dict[str, Any]:
        action = action.strip().lower()
        with self._condition:
            if action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
            elif action == "stop":
                self._stopped = True
                self._paused = False
            elif action == "message":
                if not message.strip():
                    raise ValueError("message command requires non-empty message")
                self._guidance.append(message.strip())
            else:
                raise ValueError("action must be pause, resume, stop, or message")
            self._publish_locked("control", {"action": action, "message": message})
            self._condition.notify_all()
            return self.status()

    def checkpoint(self) -> list[str]:
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait(timeout=1.0)
            if self._stopped:
                raise KeyboardInterrupt("Stopped through the interaction controller")
            guidance = list(self._guidance)
            self._guidance.clear()
            return guidance

    def ask(self, message: str, timeout: float | None = None) -> str:
        event_id = uuid.uuid4().hex
        with self._condition:
            self._publish_locked("question", {"message": message}, event_id=event_id)
            deadline = None if timeout is None else time.monotonic() + timeout
            while event_id not in self._replies and not self._stopped:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    return "(No response before timeout)"
                self._condition.wait(timeout=remaining)
            if self._stopped:
                raise KeyboardInterrupt("Stopped while waiting for human input")
            return self._replies.pop(event_id)

    def reply(self, event_id: str, message: str) -> None:
        with self._condition:
            if not any(
                event["id"] == event_id and event["kind"] == "question"
                for event in self._events
            ):
                raise ValueError(f"unknown question event: {event_id}")
            self._replies[event_id] = message.strip() or "(No input)"
            self._publish_locked("reply", {"question_id": event_id, "message": message})
            self._condition.notify_all()

    def events(self, after_seq: int = 0) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(event) for event in self._events if event["seq"] > after_seq]

    def publish(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            return self._publish_locked(kind, payload)

    def _publish_locked(
        self,
        kind: str,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "seq": self._sequence,
            "id": event_id or uuid.uuid4().hex,
            "kind": kind,
            "timestamp": time.time(),
            "payload": payload,
        }
        self._events.append(event)
        return dict(event)
