# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any, Tuple
from .base_parser import BaseParser
from ..tools import BaseTool

_logger = getLogger(f'sr_agent.{__name__}')


@BaseParser.register('json')
class JSONParser(BaseParser):
    """基于 JSON 格式的工具调用解析器。

    期望 LLM 使用以下格式输出：
    ```json
    {
        "actions": [
            {"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}
        ]
    }
    ```
    """

    def format_tools(self) -> str:
        """将工具列表格式化为 LLM 可读的描述字符串。

        Returns:
            格式化后的工具描述字符串。
        """
        lines = ["## Available Tools:", ""]

        for tool in self.tools:
            name = tool['name']
            desc = tool['description']

            # 获取工具类的签名
            try:
                if (tool_cls := BaseTool.REGISTRY_DICT.get(name)):
                    sig = inspect.signature(tool_cls.execute)
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param_name in ('self', 'args', 'kwargs'):
                            continue
                        param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any'
                        if param.default is inspect.Parameter.empty:
                            params.append(f'"{param_name}": {param_type}')
                        else:
                            params.append(f'"{param_name}": {param_type} = {param.default}')
                    signature = "{" + ", ".join(params) + "}"
                else:
                    signature = "{}"
            except Exception:
                signature = "{}"

            lines.append(f"### {name}")
            lines.append(f"- **Description**: {desc}")
            lines.append(f"- **Params**: `{signature}`")
            lines.append("")

        lines.append("## Output Format:")
        lines.append("")
        lines.append("Respond with a JSON object in the following format:")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "actions": [')
        lines.append('    {"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}')
        lines.append('  ]')
        lines.append("}")
        lines.append("```")
        lines.append("")
        lines.append("You can include multiple actions in the actions array.")

        return "\n".join(lines)

    def parse_response(self, response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """从 LLM 响应中解析工具调用。

        Args:
            response: LLM 的原始响应文本。

        Returns:
            工具调用列表，每个元素为 (tool_name, params) 元组。
        """
        actions = []

        # 尝试从响应中提取 JSON
        if (json_match := re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)):
            json_str = json_match.group(1)
        else:
            # 如果没有代码块标记，尝试直接解析整个响应
            json_str = response.strip()

        try:
            data = json.loads(json_str)
            if 'actions' in data and isinstance(data['actions'], list):
                for action in data['actions']:
                    if (tool_name := action.get('tool', action.get('name'))):
                        params = action.get('params', action.get('arguments', {}))
                        actions.append((tool_name, params))
        except json.JSONDecodeError as e:
            _logger.warning(f"Failed to parse JSON: {e}")

        return actions
