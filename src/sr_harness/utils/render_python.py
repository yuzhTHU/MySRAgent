# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

def render_python(text: str, width=120, highlight_lines=[], theme="staroffice") -> str:
    console = Console(record=True, width=width)
    with console.capture() as capture:
        code = Syntax(
            text,
            "python",
            theme=theme,
            word_wrap=True,
            line_numbers=True,
            highlight_lines=highlight_lines,
        )
        console.print(code)
    return capture.get()
