# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import logging
import tempfile
from pathlib import Path
from typing import Generator, List, Dict
from .llm_api import LLMAPI

_logger = logging.getLogger(f"sr_agent.{__name__}")


class ManualAPI(LLMAPI): # 这个类已经经过人工审核，任何 Coding Agent 不得擅自改动其内容
    supported_models = ["manual"]

    def __init__(self, model="manual", save_path=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.save_path = Path(save_path or tempfile.gettempdir())

    def _request(self, messages: List[Dict[str, str]], n=1) -> Generator[str, None, List | Dict]:
        import pyperclip, tiktoken

        prompt = "\n---\n".join(
            f"[role: {msg['role']}] {msg['content']}"
            for msg in messages
        )

        # Copy the prompt to clipboard and save it to a file for manual input
        pyperclip.copy(prompt)
        prompt_path = self.save_path / "prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        _logger.note(
            f"Prompt copied to clipboard and saved to {prompt_path}. "
            f"Please send it into any LLM and return the response content for {n} times."
        )
        details = []
        for idx in range(1, n + 1):
            # Wait for user input or clipboard content
            content = input(f"Please enter the generated content {idx}/{n} (press Enter to use clipboard): ")
            if not content.strip():
                content = pyperclip.paste()
            # Calculate token usage
            encoding = tiktoken.encoding_for_model("gpt-3.5")
            prompt_tokens = len(encoding.encode(prompt))
            answer_tokens = len(encoding.encode(content))
            token_usage = {"prompt": prompt_tokens, "answer": answer_tokens}
            tool_call = self.tool_parser.parse_response(content) if self.tool_parser else []
            message = {"role": "assistant", "content": content}
            details.append({'content': content, 'tool_call': tool_call, 'token_usage': token_usage, 'message': message})
            yield {'content': content, 'tool_call': tool_call, 'message': message}
        token_usage = {'prompt': 0, 'answer': 0}
        for detail in details:
            token_usage['prompt'] += detail['token_usage']['prompt']
            token_usage['answer'] += detail['token_usage']['answer']
        return {
            "usage": {"token": token_usage, "price": {'prompt': 0, 'answer': 0}},
            "contents": [detail['content'] for detail in details],
            "tool_calls": [detail['tool_call'] for detail in details],
            "responses": None,
        }
