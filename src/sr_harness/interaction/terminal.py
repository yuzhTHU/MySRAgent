"""Terminal interaction manager."""
from __future__ import annotations

import re

from ..utils import render_markdown, tag2ansi
from .manager import InteractionManager


class TerminalInteractionManager(InteractionManager):
    """Read human guidance from the current terminal."""

    _SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

    def ask_human(self, message: str) -> str:
        from prompt_toolkit import prompt
        from prompt_toolkit.patch_stdout import patch_stdout

        print(tag2ansi(f"\n[gray]{'=' * 60}[reset]"))
        print(tag2ansi("[red bold][Agent asks for guidance][reset]"))
        print(tag2ansi(f"[gray]{'-' * 60}[reset]"))
        print(tag2ansi(render_markdown(message)))
        print(tag2ansi(f"[gray]{'=' * 60}[reset]"))
        with patch_stdout():
            response = prompt("Your response (press Enter to let agent continue): ")
        return self._SURROGATE_RE.sub("", response.strip() or "(No input)")
