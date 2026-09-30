# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any, Tuple
from .base_parser import BaseParser
from ..api.core import ToolCall
from ..tools import BaseTool

_logger = getLogger(f'sr_agent.{__name__}')


@BaseParser.register('text')
class TextParser(BaseParser):
    """基于文本格式的工具调用解析器。

    期望 LLM 使用以下格式输出：
    Action: tool_name(param1=value1, param2=value2)
    """

    def format_tools(self) -> str:
        """Format tool list into a description string for LLM.

        Returns:
            Formatted tool description string.
        """
        lines = ["## Available Tools:", ""]

        for tool in self.tools:
            name = tool['name']
            desc = tool['description']

            # 获取工具类的签名和 docstring
            try:
                if (tool_cls := BaseTool.create(name, create_instance=False)):
                    # 获取 execute 方法的签名
                    sig = inspect.signature(tool_cls.execute)
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param_name in ('self', 'args', 'kwargs'):
                            continue
                        # 构建带类型注解的参数描述
                        param_parts = []
                        param_parts.append(param_name)
                        # 添加类型注解
                        if param.annotation is not inspect.Parameter.empty:
                            type_hint = self._format_type(param.annotation)
                            param_parts.append(f": {type_hint}")
                        # 添加默认值
                        if param.default is not inspect.Parameter.empty:
                            default_val = param.default
                            param_parts.append(f" = {default_val!r}")
                        params.append("".join(param_parts))
                    signature = f"{name}({', '.join(params)})"

                    # 获取 docstring
                    docstring = inspect.getdoc(tool_cls.execute) or "No detailed documentation available."
                else:
                    signature = f"{name}(...)"
                    docstring = "No detailed documentation available."
            except Exception:
                signature = f"{name}(...)"
                docstring = "No detailed documentation available."

            lines.append(f"### {name}")
            lines.append(f"- **Description**: {desc}")
            lines.append(f"- **Signature**: `{signature}`")
            lines.append(f"- **DocString**: {docstring}")
            lines.append("")

        lines.append("## Output Format:")
        lines.append("")
        lines.append("Use the following format for tool calls:")
        lines.append("")
        lines.append("Action: tool_name(param1=value1, param2=value2)")
        lines.append("")
        lines.append("You can make multiple tool calls in sequence, one Action per line.")

        return "\n".join(lines)

    def parse_response(self, response: str) -> List[ToolCall]:
        """从 LLM 响应中解析工具调用。

        Args:
            response: LLM 的原始响应文本。

        Returns:
            工具调用列表。
        """
        tool_calls = []
        for line in response.strip().splitlines():
            line = line.strip()
            if line.startswith('Action:'):
                action_line = line.removeprefix('Action:').strip()
                if (match := re.match(r'(\w+)\s*\((.*)\)', action_line)):
                    tool_name = match.group(1)
                    if (params_str := match.group(2).strip()):
                        params = self._parse_params(params_str)
                    else:
                        params = {}
                    tool_calls.append(ToolCall(name=tool_name, params=params, raw_str=line))
                else:
                    _logger.warning(f"Failed to parse action line: '{line}'")
        return tool_calls

    def format_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """将工具调用列表格式化为字符串，供 LLM 参考。

        Args:
            tool_calls: 工具调用列表。

        Returns:
            格式化后的工具调用字符串。
        """
        lines = []
        for tool_call in tool_calls:
            params_str = ', '.join(f"{k}={v!r}" for k, v in tool_call.params.items())
            lines.append(f"Action: {tool_call.name}({params_str})")
        return "\n".join(lines)

    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """解析参数字符串为字典。

        Args:
            params_str: 参数字符串，如 'x_vars=["phi0", "phi1"], y_var="y"'

        Returns:
            参数字典
        """
        try:
            # 先尝试 JSON 解析（如果整个字符串是 JSON 对象）
            if params_str.startswith('{') and params_str.endswith('}'):
                return json.loads(params_str)
            else:
                # 将 key=value, key2=value2 格式转换为字典
                params_dict = {}
                parts = self._split_params(params_str)
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip().lower()  # 转换为小写
                        value = value.strip()
                        try:
                            params_dict[key] = literal_eval(value)
                        except (ValueError, SyntaxError):
                            params_dict[key] = value
                return params_dict
        except Exception as e:
            _logger.warning(f"Failed to parse params '{params_str}': {e}")
            return {'raw_params': params_str}

    def _format_type(self, annotation) -> str:
        """Format type annotation into a readable string.

        Args:
            annotation: The type annotation to format.

        Returns:
            Formatted type string.
        """
        import typing
        if annotation is inspect.Parameter.empty:
            return ""

        # 处理常见类型
        if annotation is type(None):
            return "None"
        if annotation is ...:
            return "..."

        # 处理 typing 模块的类型
        origin = getattr(annotation, '__origin__', None)
        if origin is not None:
            # 处理 Optional[T] = Union[T, None]
            if origin is typing.Union:
                args = getattr(annotation, '__args__', ())
                if type(None) in args:
                    non_none = [t for t in args if t is not type(None)]
                    if len(non_none) == 1:
                        return f"Optional[{self._format_type(non_none[0])}]"
            # 处理 List[T], Dict[K, V], Tuple[...]
            args = getattr(annotation, '__args__', ())
            args_str = ', '.join(self._format_type(arg) for arg in args)
            name = getattr(origin, '__name__', str(origin))
            return f"{name}[{args_str}]"

        # 处理普通类型
        if hasattr(annotation, '__name__'):
            return annotation.__name__

        return str(annotation)

    def _split_params(self, params_str: str) -> List[str]:
        """分割参数字符串，处理引号内的逗号。

        Args:
            params_str: 参数字符串，如 'x_vars=["phi0", "phi1"], y_var="y"'

        Returns:
            分割后的参数列表
        """
        parts = []
        current = []
        depth = 0
        in_string = False
        string_char = None

        for char in params_str:
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current.append(char)
            elif char in '[{(':
                depth += 1
                current.append(char)
            elif char in ']})':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0 and not in_string:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            parts.append(''.join(current).strip())

        return parts
