# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import logging
from typing import Generator, Tuple, List, Dict
from .llm_api import LLMAPI

_logger = logging.getLogger(__name__)


class ManualAPI(LLMAPI):
    supported_models = ["manual"]

    def _request(self, prompt: str) -> Generator[Tuple[str, dict], None, List | Dict]:
        raise DeprecationWarning
        import pyperclip
        import tiktoken

        # Copy the prompt to clipboard and save it to a file for manual input
        pyperclip.copy(prompt)
        save_path = self.save_path or "."
        prompt_path = f"{save_path}/manual_prompt.md"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        _logger.note(
            f"Prompt copied to clipboard (and saved to {prompt_path}). Please paste it into the LLM interface and return the generated program."
        )
        results = []
        for idx in range(1, self.generate_per_prompt + 1):
            # Wait for user input or clipboard content
            program = input(
                f"Please enter the generated program {idx}/{self.generate_per_prompt} (press Enter to use clipboard): "
            )
            if not program.strip():
                program = pyperclip.paste()
            # Calculate token usage
            encoding = tiktoken.encoding_for_model("gpt-3.5")
            prompt_tokens = len(encoding.encode(prompt))
            answer_tokens = len(encoding.encode(program))
            total_tokens = prompt_tokens + answer_tokens
            usage = {
                "total": total_tokens,
                "prompt": prompt_tokens,
                "answer": answer_tokens,
            }
            results.append(program)
            yield program, usage
        return {"prompt": prompt, "response": results}
