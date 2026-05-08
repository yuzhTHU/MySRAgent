# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工具基类定义。

所有工具都应继承自 BaseTool，并提供统一的接口。
"""
from __future__ import annotations

import time
import traceback
import numpy as np
import nd2py as nd
from logging import getLogger
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from docstring_parser import DocstringStyle, parse
from abc import ABC, abstractmethod
from types import NoneType, UnionType
from inspect import Parameter, signature
from typing import Any, Dict, Literal, Union, get_args, get_origin, get_type_hints
from ..utils import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')


@dataclass
class ToolMetadata:
    """工具元数据。

    Attributes:
        name: 工具名称，用于 LLM 识别和调用。
        description: 工具简述，说明工具的功能和适用场景。
            设置为 None 以从 execute 方法的 docstring 中自动提取工具描述。
        parameters: OpenAI/OpenRouter function calling 兼容的 JSON Schema。
            设置为 None 以从 execute 方法的签名和 docstring 中自动推断基础 schema。
    """
    name: str
    description: str | None = None
    parameters: Dict[str, Any] | None = None


@dataclass
class ToolCallResult:
    """工具调用结果。

    Attributes:
        ok: 是否成功执行工具（当且仅当出现无法处理的报错时为 False）
        result: 运行结果, 用于存档和后续分析
        result_str: 对 result 格式化后的版本, 用于展示给 LLM 的结果字符串
        meta_data: 额外的元信息，如执行时间、日志等
    """
    ok: bool
    result: Dict[str, Any]
    result_str: str
    meta_data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access to the wrapped result for legacy callers."""
        return self.result.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.result[key]


class BaseTool(ABC, FactoryMixin):
    """工具基类。所有工具都应继承此类，并设置 / 实现以下字段和方法：
    - metadata: ToolMetadata 实例，提供工具的名称、描述和参数 schema（若不提供则尝试自动推断）
    - execute(): 工具的核心执行方法，接受 LLM 生成的参数并返回结果字典。工具的 execute 方法应该尽量保持参数简单，复杂的上下文信息（如数据）可以通过工具实例的 context 属性传入。
    - format_result_dict(): 可选的类方法，用于将 execute 的结果字典格式化为字符串，供 LLM 阅读。默认实现是直接转换为字符串，不同工具可以根据需要重写此方法以提供更友好的结果展示。
    """
    metadata: ToolMetadata = None

    def __init_subclass__(cls, **kwargs):
        """在子类定义时自动设置元数据"""
        super().__init_subclass__(**kwargs)
        if cls.metadata is None:
            cls.metadata = ToolMetadata()
        if cls.metadata.description is None:
            cls.metadata.description = cls.infer_tool_description()
        if cls.metadata.parameters is None:
            cls.metadata.parameters = cls.infer_tool_parameters()

    def __init__(self, **context):
        """ context 中传入一些工具执行时需要的上下文信息，如数据、模型等，这些信息不适合放在 execute 的参数列表中让 LLM 生成 """
        self.context = context

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """ 执行工具并返回运行结果
        1) 这个方法的参数列表需要由 LLM 生成，因此其参数应该尽量简单
        2) 复杂的上下文信息（如数据）可以通过工具实例的 context 属性传入
        3) 在实现时需要注意，此工具可能被多个进程/线程并行调用，需要保证线程安全
        4) execute 方法的 docstring 将被用于解析生成 metadata.description 和 metadata.parameters
            docstring 中 Args: 之前的部分被用于生成 metadata.description
            Args: 之后的部分被用于生成 metadata.parameters 的 description 字段
        5) execute 方法的 signature 和 type hints 将被用于解析生成 metadata.parameters 的 schema 字段
            目前支持 int / float / str / bool / List / Dict 以及它们的组合
            不支持 *args 和 **kwargs 这类不定参数，也不支持 Optional / Union / Literal 等复杂类型的自动解析
            对于复杂的参数类型，建议直接在 ToolMetadata.parameters 中手动指定完整的 JSON Schema
        """
        pass

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """将工具执行结果格式化为字符串，供 LLM 阅读。不同工具可以根据需要重写此方法以提供更友好的结果展示。

        Args:
            result: 工具执行结果字典。
        """
        return str(result)


    def __call__(self, *args, **kwargs) -> ToolCallResult:
        """工具调用入口"""
        start_time = time.time()
        try:
            result = self.execute(*args, **kwargs)
            result_str = self.format_result_dict(result)
            meta_data = {
                "timestamp": start_time,
                "execution_time": time.time() - start_time, 
                "tool": self.metadata.name
            }
            return ToolCallResult(ok=True, result=result, result_str=result_str, meta_data=meta_data)
        except Exception as e:
            error_msg = f"Error executing {self.metadata.name}: [{type(e).__name__}] {e}\n{traceback.format_exc()}"
            meta_data = {
                "timestamp": start_time,
                "execution_time": time.time() - start_time, 
                "tool": self.metadata.name
            }
            _logger.error(error_msg)
            return ToolCallResult(ok=False, result={'error': error_msg}, result_str=error_msg, meta_data=meta_data)

    @classmethod
    def to_tool_list(cls, tools_used: list[str] | None = None) -> list[dict]:
        """加载 OpenRouter/OpenAI 兼容的 tools 定义。

        Args:
            tools_used: 可用工具名列表。None 表示使用全部已注册工具。

        Returns:
            形如 ``[{"type": "function", "function": {...}}]`` 的工具定义列表。
        """
        tool_list = []
        for name, tool_cls in cls.REGISTRY_DICT.items():
            if tools_used is None or tool_cls.metadata.name in tools_used:
                tool_list.append(tool_cls.to_dict())
        return tool_list

    @classmethod
    def load_tool_list(cls, tools_used: list[str] | None = None) -> list[dict]:
        """加载兼容 legacy parser 的工具元数据列表。"""
        tool_list = []
        for name, tool_cls in cls.REGISTRY_DICT.items():
            if tools_used is None or tool_cls.metadata.name in tools_used:
                tool_list.append({
                    "name": tool_cls.metadata.name,
                    "description": tool_cls.metadata.description,
                    "parameters": tool_cls.metadata.parameters,
                })
        return tool_list

    @classmethod
    def load_tool_classes(cls, tools_used: list[str] | None = None) -> list[type["BaseTool"]]:
        """加载工具类列表，供 LLM API 和 native function calling 使用。"""
        tool_list = []
        for name, tool_cls in cls.REGISTRY_DICT.items():
            if tools_used is None or tool_cls.metadata.name in tools_used:
                tool_list.append(tool_cls)
        return tool_list

    @classmethod
    def to_dict(cls) -> dict:
        """导出 OpenRouter/OpenAI function calling 工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": cls.metadata.name, 
                "description": cls.metadata.description, 
                "parameters": cls.metadata.parameters
            },
        }

    @classmethod
    def infer_tool_description(cls) -> str:
        """从 execute 方法的 docstring (不含 ARGS 和 RETURNS) 自动提取工具描述。"""
        doc = getattr(cls.execute, "__doc__", None) or ""
        description_lines = []
        for line in doc.splitlines():
            line = line.strip()
            if line == "Args:":
                break
            if line:
                description_lines.append(line)
        return "\n".join(description_lines) if description_lines else "(no description provided)"
    
    @classmethod
    def infer_tool_parameters(cls) -> Dict[str, Any]:
        """从 execute 方法的 signature 和 docstring 自动推断 parameters schema。
        注：该推断只覆盖常用 Python/typing 类型。复杂约束、枚举说明和更精确的格式建议直接
        写入 ``ToolMetadata.parameters``，避免让 LLM 猜测参数含义。
        """
        properties = {}
        required = []

        try:
            sig = signature(cls.execute)
            type_hints = get_type_hints(cls.execute)
            descriptions = cls.parse_args_docstring(cls.execute)
        except Exception as e:
            _logger.warning(f"Failed to parse signature or type hints for {cls.__name__} since [{type(e).__name__}] {e}")
            return {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                _logger.warning(
                    f"Parameter '{param_name}' in {cls.__name__}.execute is a variable positional or keyword parameter, "
                    f"which is not supported for automatic schema inference and will be ignored. "
                    f"Please specify its schema manually in ToolMetadata.parameters if needed."
                )
                continue

            schema = {}
            annotation = type_hints.get(param_name, param.annotation) # e.g., List[int], Optional[str], etc.
            schema |= cls.parse_args_typehints(annotation) # e.g., {"type": "array", "items": {"type": "integer"}} for List[int]
            schema |= {'description': descriptions[param_name]}
            if param.default is not Parameter.empty:
                schema |= {'default': param.default}
            else:
                required.append(param_name)
            properties[param_name] = schema

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @staticmethod
    def parse_args_docstring(func: callable) -> dict[str, str]:
        """从函数 docstring 中提取参数说明。"""
        docstring = getattr(func, "__doc__", None)
        parsed = parse(docstring, style=DocstringStyle.GOOGLE)
        description_dict = {}
        for param in parsed.params:
            if param.description is not None:
                description_dict[param.arg_name] = param.description
        set1 = set(description_dict.keys())
        set2 = set(signature(func).parameters.keys()) - {"self"}
        if set1 - set2:
            _logger.warning(
                f"Descriptions found for parameters {set1 - set2} in {func.__qualname__} docstring, "
                f"but these parameters are not in the function signature. Please check for typos or remove these descriptions."
            )
        if set2 - set1:
            _logger.warning(
                f"Parameters {set2 - set1} in {func.__qualname__} signature do not have descriptions in the docstring. "
                f"Please add descriptions for better LLM understanding."
            )
            description_dict = description_dict | {param: "(no description provided)" for param in set2 - set1}
        return description_dict

    @classmethod
    def parse_args_typehints(cls, annotation: Any) -> Dict[str, Any]:
        """将常见 Python 类型注解转换为 JSON Schema。"""
        if annotation is Parameter.empty or annotation is Any:
            return {}

        origin = get_origin(annotation) # e.g., list, dict, Union, etc.
        args = get_args(annotation) # e.g., (int,) for List[int], (str, NoneType) for Optional[str], etc.

        if origin is Literal:
            values = list(args)
            schema = {"enum": values}
            if non_none_values := [value for value in values if value is not None]:
                schema["type"] = cls.parse_json_type(type(non_none_values[0]))
            return schema

        if origin is list:
            item_type = args[0] if args else Any
            return {"type": "array", "items": cls.parse_args_typehints(item_type)}

        if origin is tuple:
            if not (item_schemas := [cls.parse_args_typehints(arg) for arg in args]):
                return {"type": "array", "items": {}}
            elif args[-1] is not Ellipsis:
                return {
                    "type": "array",
                    # Draft 7 tuple validation uses an array-valued items.
                    # Avoid prefixItems for broader OpenRouter/OpenAI compatibility.
                    "items": item_schemas,
                    "minItems": len(item_schemas),
                    "maxItems": len(item_schemas),
                }
            else:
                return {"type": "array", "items": item_schemas[0]}

        if origin is dict:
            return {"type": "object"}

        if origin in (Union, UnionType):
            schemas = [
                {"type": "null"} if arg is NoneType else cls.parse_args_typehints(arg)
                for arg in args
            ]
            return {"anyOf": schemas}

        if json_type := cls.parse_json_type(annotation):
            return {"type": json_type}

        return {}

    @staticmethod
    def parse_json_type(value_type: Any) -> str:
        """根据 Python 字面值推断 JSON Schema type。"""
        if value_type is bool:
            return "boolean"
        if value_type is int:
            return "integer"
        if value_type is float:
            return "number"
        if value_type is str:
            return "string"
        if value_type is list:
            return "array"
        if value_type is dict:
            return "object"
        return ""

    def evaluate(self, eq: str = None, y_pred: np.ndarray = None, y_true: np.ndarray = None) -> Dict[str, float]:
        """Evaluate predictions or a formula against the target in context.

        When ``eq`` is provided, the formula is evaluated with variables from
        ``self.context["data"]`` and target name ``self.context["target"]``.
        Legacy contexts using ``x`` and ``y`` are also supported.
        """
        if eq is not None:
            data, target = self.context['data'], self.context['target']
            f = nd.parse(eq.replace("^", "**"))
            y_pred = f.eval(data)
            y_true = data[target]
        elif y_pred is None or y_true is None:
            raise ValueError("Either eq or both y_pred and y_true must be provided.")

        y_pred = y_pred + 0 * y_true # broadcast to target shape if needed

        residuals = y_pred - y_true
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))
        if np.any(non_zero := ~np.isclose(y_true, 0.0)):
            mape = float(np.mean(np.abs(residuals[non_zero] / y_true[non_zero])))
        else:
            mape = 0.0 if np.allclose(y_pred, y_true) else float("inf")
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        finite = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred_finite = y_pred[finite]
        y_true_finite = y_true[finite]
        if np.count_nonzero(finite) < 2:
            pearson_r = float("nan")
            spearman_r = float("nan")
        elif np.std(y_pred_finite) == 0 or np.std(y_true_finite) == 0:
            pearson_r = float("nan")
            spearman_r = float("nan")
        else:
            pearson_r = float(np.corrcoef(y_true_finite, y_pred_finite)[0, 1])
            spearman_r = float(stats.spearmanr(y_true_finite, y_pred_finite).statistic)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
        }
