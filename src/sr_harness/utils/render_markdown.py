# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax


def render_markdown(text: str, width=120, theme="staroffice") -> str:
    console = Console(record=True, width=width)
    with console.capture() as capture:
        matches = list(re.finditer(r"```(\w+)?\n(.*?)\n```", text, re.DOTALL))
        current = 0
        for match in matches:
            lang, code = match.groups()
            start, end = match.span()
            md = Markdown(text[current:start])
            code = Syntax(code, lang or "text", word_wrap=True, line_numbers=True, theme=theme)
            console.print(md)
            console.print(code)
            current = end
        if current < len(text):
            console.print(Markdown(text[current:]))
    return capture.get()
