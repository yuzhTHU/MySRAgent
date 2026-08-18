# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工具基类定义。

所有工具都应继承自 BaseTool，并提供统一的接口。
"""
from __future__ import annotations

import time
import traceback
import warnings
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
from typing import Any, Dict, List, Literal, Union, get_args, get_origin, get_type_hints
from ..utils import FactoryMixin, log_exception

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


class ToolRunAbort(RuntimeError):
    """Raise from a tool to bypass BaseTool.__call__ error handling."""


class BaseTool(ABC, FactoryMixin):
    """工具基类。所有工具都应继承此类，并设置 / 实现以下字段和方法：
    - metadata: ToolMetadata 实例，提供工具的名称、描述和参数 schema（若不提供则尝试自动推断）
    - execute(): 工具的核心执行方法，接受 LLM 生成的参数并返回结果字典。工具的 execute 方法应该尽量保持参数简单，复杂的上下文信息（如数据）可以通过工具实例的 context 属性传入。
    - format_result_dict(): 可选的类方法，用于将 execute 的结果字典格式化为字符串，供 LLM 阅读。默认实现是直接转换为字符串，不同工具可以根据需要重写此方法以提供更友好的结果展示。
    """
    metadata: ToolMetadata = None
    MAX_FORMULA_LENGTH = 10000
    MAX_RESULT_STR_LENGTH = 64 * 1024

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
            if not isinstance(result, dict):
                _logger.critical(f"Tool {self.metadata.name} execute() should return a dict, but got {type(result)}, please check the implementation.")
            result_str = self.format_result_dict(result)
            result_str = self.truncate_result_str(result_str)
            meta_data = {
                "timestamp": start_time,
                "execution_time": time.time() - start_time, 
                "tool": self.metadata.name
            }
            return ToolCallResult(ok=True, result=result, result_str=result_str, meta_data=meta_data)
        except ToolRunAbort:
            raise
        except Exception as e:
            error_msg = f"Error executing {self.metadata.name}: {log_exception(e)}"
            error_msg = self.truncate_result_str(error_msg)
            meta_data = {
                "timestamp": start_time,
                "execution_time": time.time() - start_time, 
                "tool": self.metadata.name
            }
            _logger.error(error_msg)
            return ToolCallResult(ok=False, result={'error': error_msg}, result_str=error_msg, meta_data=meta_data)

    @classmethod
    def normalize_formula(cls, eq: str, *, strip_modules: bool = True) -> str:
        """Validate and normalize an agent-provided formula before parsing it."""
        if not isinstance(eq, str):
            raise TypeError(f"Formula must be a string, got {type(eq).__name__}.")
        eq = eq.strip()
        if len(eq) > cls.MAX_FORMULA_LENGTH:
            raise ValueError(
                f"Formula is too long: {len(eq)} characters; "
                f"maximum allowed is {cls.MAX_FORMULA_LENGTH}."
            )
        eq = eq.replace("^", "**")
        if strip_modules:
            eq = eq.replace("np.", "").replace("numpy.", "").replace("math.", "")
        return eq

    @classmethod
    def parse_formula(cls, eq: str) -> nd.Symbol:
        """Normalize and parse a formula with the constants supported by all tools."""
        return nd.parse(
            cls.normalize_formula(eq),
            variables={"pi": np.pi, "e": np.e},
        )

    @classmethod
    def truncate_result_str(cls, text: str) -> str:
        """Bound text sent back to the LLM while preserving the full raw result."""
        text = str(text)
        limit = cls.MAX_RESULT_STR_LENGTH
        if len(text) <= limit:
            return text
        suffix = "... (characters ignored)"
        keep = max(0, limit - len(suffix) - 32)
        ignored = len(text) - keep
        suffix = f"... ({ignored} characters ignored)"
        keep = max(0, limit - len(suffix))
        ignored = len(text) - keep
        suffix = f"... ({ignored} characters ignored)"
        return text[: max(0, limit - len(suffix))] + suffix

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
        args = get_args(annotation) #  e.g., (int,) for List[int], (str, NoneType) for Optional[str], etc.

        if origin is Literal:
            values = list(args)
            schema = {"enum": values}
            if non_none_values := [v for v in values if v is not None]:
                schema["type"] = cls.parse_json_type(type(non_none_values[0]))
            return schema
        elif origin is list:
            return {"type": "array", "items": cls.parse_args_typehints(args[0] if args else Any)}
        elif origin is dict:
            return {"type": "object"}
        elif origin is tuple:
            # Tuple without type args
            if not args:
                return {"type": "array", "items": {}}
            # Variable-length tuple, e.g. Tuple[str, ...]
            elif args[-1] is Ellipsis:
                return {"type": "array", "items": cls.parse_args_typehints(args[0] if args else Any)}
            item_schemas = [cls.parse_args_typehints(arg) for arg in args]
            # Homogeneous fixed tuple, e.g. Tuple[str, str]
            if all(schema == item_schemas[0] for schema in item_schemas):
                return {"type": "array", "items": item_schemas[0], "minItems": len(args), "maxItems": len(args)}
            # Heterogeneous fixed tuple, e.g. Tuple[str, int]
            else:
                # This is less position-strict than prefixItems, but more compatible with OpenAI/Azure/OpenRouter tool schemas.
                # It means each item may match any of the tuple element schemas, while minItems/maxItems still enforce tuple length.
                _logger.warning(
                    f"Tuple parameter with heterogeneous types detected in {cls.__name__}.execute. "
                    f"This is not recommended since it may lead to less precise schema and LLM confusion. "
                    f"Please consider using a more specific schema or adding descriptions to clarify the expected format."
                )
                return {"type": "array", "items": {"anyOf": item_schemas}, "minItems": len(args), "maxItems": len(args)}
        elif origin in (Union, UnionType):
            schemas = [{"type": "null"} if arg is NoneType else cls.parse_args_typehints(arg) for arg in args]
            return {"anyOf": schemas}
        elif json_type := cls.parse_json_type(annotation):
            return {"type": json_type}
        else:
            _logger.warning(
                f"Unsupported type annotation '{annotation}' in {cls.__name__}.execute. "
                f"Automatic schema inference will not be able to parse this type, and it will be treated as an untyped parameter. "
                f"Please consider using basic types (int, float, str, bool) or common generics (List, Dict) for better compatibility, "
                f"or directly specify the parameter schema in ToolMetadata.parameters if a more complex structure is needed."
            )
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

    @classmethod
    def calculate_metrics(cls, f: nd.Symbol, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics for already-computed target and prediction arrays.

        This low-level helper is intentionally separate from :meth:`evaluate` so
        code-defined models can supply predictions produced outside nd2py while
        still using exactly the same metric definitions.
        """
        try:
            y_pred, y_true = np.broadcast_arrays(np.asarray(y_pred), np.asarray(y_true))
        except ValueError as exc:
            raise ValueError(
                f"Prediction shape {np.shape(y_pred)} and target shape {np.shape(y_true)} "
                "cannot be broadcast to a common shape."
            ) from exc
        y_pred = y_pred.astype(float, copy=False).flatten()
        y_true = y_true.astype(float, copy=False).flatten()

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
        n_samples = int(y_true.size)
        n_parameters = sum(np.size(op.value) for op in f.iter_preorder() if isinstance(op, nd.Number))
        if np.isfinite(ss_res) and ss_res > 0:
            log_likelihood = -n_samples / 2 * (
                np.log(2 * np.pi) + np.log(ss_res / n_samples) + 1
            )
            aic = float(2 * n_parameters - 2 * log_likelihood)
            bic = float(n_parameters * np.log(n_samples) - 2 * log_likelihood)
        elif ss_res == 0:
            aic = bic = float("-inf")
        else:
            aic = bic = float("nan")

        pearson_r, spearman_r = cls.correlation_coefficients(y_true, y_pred)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "aic": aic,
            "bic": bic,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "complexity": len(f),
        }

    def evaluate(
        self,
        f: nd.Symbol,
        y: nd.Symbol,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a symbolic prediction against a symbolic target.

        ``f`` and ``y`` define the prediction and target.
        Formula complexity is computed as ``len(f)``.

        Set ``show_diagnostics`` to include a compact residual error profile,
        the worst samples, and the strongest residual-variable correlations.
        """
        if not isinstance(f, nd.Symbol) or not isinstance(y, nd.Symbol):
            raise TypeError("f and y must both be nd2py.Symbol instances.")

        data_split_results = {
            'train': {'metrics': None, 'diagnostics': None},
            'validation': {'metrics': None, 'diagnostics': None},
        }
        if data := self.context.get("data"):
            y_pred = f.eval(data)
            y_true = y.eval(data)
            data_split_results['train']['metrics'] = self.calculate_metrics(f, y_true, y_pred)
            if show_diagnostics:
                data_split_results['train']['diagnostics'] = self.residual_diagnostics(
                    y_true=np.asarray(y_true),
                    y_pred=np.asarray(y_pred),
                    data=data,
                    target_expression=y.to_str(),
                )
            else:
                data_split_results['train'].pop('diagnostics')
        else:
            raise ValueError("Training data is missing in context['data'] for evaluation.")
        if data := self.context.get("evaluation_data"):
            y_pred = f.eval(data)
            y_true = y.eval(data)
            data_split_results['validation']['metrics'] = self.calculate_metrics(f, y_true, y_pred)
            if show_diagnostics:
                data_split_results['validation']['diagnostics'] = self.residual_diagnostics(
                    y_true=np.asarray(y_true),
                    y_pred=np.asarray(y_pred),
                    data=data,
                    target_expression=y.to_str(),
                )
            else:
                data_split_results['validation'].pop('diagnostics')
        else:
            data_split_results.pop('validation')

        target = self.context["target"]
        var_names = {var.name for var in f.iter_preorder() if isinstance(var, nd.Variable)}
        is_candidate = (y.to_str() == target) and (target not in var_names)
        evaluation = {
            "formula": f.to_str(),
            "is_candidate": is_candidate,
            "data_split_results": data_split_results,
        }
        return evaluation

    def failed_evaluation(self, formula: str = "(None)", show_diagnostics: bool = True) -> Dict[str, Any]:
        """Build the common result shape when a formula-producing backend fails."""
        empty_metrics = {
            "mse": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "mape": float("inf"),
            "r2": -float("inf"),
            "aic": float("inf"),
            "bic": float("inf"),
            "pearson_r": -float("inf"),
            "spearman_r": -float("inf"),
            "complexity": float("inf"),
        }
        data_split_results = {
            'train': {'metrics': None},
            'validation': {'metrics': None},
        }
        if data := self.context.get("data"):
            data_split_results['train']['metrics'] = empty_metrics
        else:
            raise ValueError("Training data is missing in context['data'] for evaluation.")
        if data := self.context.get("evaluation_data"):
            data_split_results['validation']['metrics'] = empty_metrics
        else:
            data_split_results.pop('validation')
        return {
            "formula": formula,
            "is_candidate": False,
            "data_split_results": data_split_results,
        }

    @classmethod
    def format_evaluation_result(cls, result: Dict[str, Any], title: str = "Formula evaluation") -> str:
        """Format the common formula-evaluation schema for an LLM."""
        split_results = result["data_split_results"]
        train_result = split_results["train"]
        metrics = train_result["metrics"]
        lines = [
            f"{title}: {result['formula']}",
            f"Training-set fit quality: RMSE={metrics.get('rmse', float('nan')):.6g}, "
            f"MAE={metrics.get('mae', float('nan')):.6g}, "
            f"R²={metrics.get('r2', float('nan')):.6g}, "
            f"formula complexity={metrics.get('complexity', '?')}.",
        ]
        if validation_result := split_results.get("validation"):
            validation_metrics = validation_result["metrics"]
            lines.append(
                f"Validation-set performance: RMSE={validation_metrics.get('rmse', float('nan')):.6g}, "
                f"MAE={validation_metrics.get('mae', float('nan')):.6g}, "
                f"R²={validation_metrics.get('r2', float('nan')):.6g}."
            )
            train_rmse = metrics.get("rmse", float("nan"))
            validation_rmse = validation_metrics.get("rmse", float("nan"))
            if (np.isfinite(train_rmse) and np.isfinite(validation_rmse)
                    and validation_rmse > max(2.0 * train_rmse, train_rmse + 1e-12)):
                lines.append(
                    "Warning: validation error is substantially larger than training error; "
                    "the candidate may be overfitting or the validation domain may differ."
                )
        if diagnostics := train_result.get("diagnostics"):
            profile = diagnostics.get("error_profile", {})
            lines.append(
                f"Error extremes: 95th-percentile absolute error="
                f"{profile.get('p95_absolute_error', float('nan')):.6g}; maximum absolute error="
                f"{profile.get('max_absolute_error', float('nan')):.6g}; median signed residual="
                f"{profile.get('median_signed_residual', float('nan')):.6g}."
            )
            if correlations := diagnostics.get("strongest_residual_correlations"):
                strongest = correlations[0]
                lines.append(
                    f"Strongest remaining residual association: {strongest['variable']} "
                    f"(residual Pearson={strongest['pearson']:.4g}, residual Spearman="
                    f"{strongest['spearman']:.4g}). This can suggest missing structure, but very small "
                    "absolute residuals may make the correlation practically irrelevant."
                )
        if validation_diagnostics := split_results.get("validation", {}).get("diagnostics"):
            profile = validation_diagnostics.get("error_profile", {})
            lines.append(
                f"Validation residual profile: 95th-percentile absolute error="
                f"{profile.get('p95_absolute_error', float('nan')):.6g}; maximum absolute error="
                f"{profile.get('max_absolute_error', float('nan')):.6g}."
            )
        lines.append(
            "Eligible for submission (default target predicted without using the target as an input): "
            f"{result['is_candidate']}. This is an interface check, not proof that the formula is correct."
        )
        return "\n".join(lines)

    @classmethod
    def residual_diagnostics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        data: Dict[str, Any] = None,
        target_expression: str = None,
        max_samples: int = 3,
        max_correlations: int = 5,
    ) -> Dict[str, Any]:
        """Return compact, high-signal diagnostics for prediction residuals."""
        y_true = np.asarray(y_true, dtype=float).flatten()
        y_pred = np.asarray(y_pred, dtype=float).flatten()
        residual = y_pred - y_true
        absolute_error = np.abs(residual)
        finite_error = np.isfinite(residual)

        target_scale = float(np.median(np.abs(y_true[np.isfinite(y_true)]))) if np.any(np.isfinite(y_true)) else 0.0
        relative_floor = max(np.finfo(float).eps, target_scale * 1e-12)
        relative_valid = finite_error & np.isfinite(y_true) & (np.abs(y_true) > relative_floor)
        relative_error = np.full_like(absolute_error, np.nan, dtype=float)
        relative_error[relative_valid] = absolute_error[relative_valid] / np.abs(y_true[relative_valid])

        finite_absolute = absolute_error[finite_error]
        finite_relative = relative_error[np.isfinite(relative_error)]
        error_profile = {
            "median_signed_residual": float(np.median(residual[finite_error])) if np.any(finite_error) else float("nan"),
            "p95_absolute_error": float(np.quantile(finite_absolute, 0.95)) if finite_absolute.size else float("nan"),
            "max_absolute_error": float(np.max(finite_absolute)) if finite_absolute.size else float("nan"),
            "p95_relative_error": float(np.quantile(finite_relative, 0.95)) if finite_relative.size else float("nan"),
            "max_relative_error": float(np.max(finite_relative)) if finite_relative.size else float("nan"),
            "relative_error_defined_ratio": float(np.mean(relative_valid)),
        }

        numeric_data = {}
        for name, values in (data or {}).items():
            array = np.asarray(values)
            if array.size == y_true.size and np.issubdtype(array.dtype, np.number):
                numeric_data[name] = array.flatten()

        rank_score = np.where(finite_error, absolute_error, -np.inf)
        sample_limit = max(0, int(max_samples))
        worst_indices = (
            np.argsort(rank_score)[-sample_limit:][::-1]
            if sample_limit else np.array([], dtype=int)
        )
        worst_samples = []
        for index in worst_indices:
            if not np.isfinite(rank_score[index]):
                continue
            worst_samples.append({
                "index": int(index),
                "row": {name: float(values[index]) for name, values in numeric_data.items()},
                "y_true": float(y_true[index]),
                "y_pred": float(y_pred[index]),
                "absolute_error": float(absolute_error[index]),
                "relative_error": float(relative_error[index]),
            })

        correlation_arrays = dict(numeric_data)
        if target_expression and target_expression not in correlation_arrays:
            correlation_arrays[target_expression] = y_true
        correlations = []
        for name, values in correlation_arrays.items():
            pearson, spearman = cls.correlation_coefficients(values, residual)
            strength = max(abs(pearson), abs(spearman))
            if np.isfinite(strength):
                correlations.append({
                    "variable": name,
                    "pearson": pearson,
                    "spearman": spearman,
                    "strength": strength,
                })
        correlations.sort(key=lambda item: item["strength"], reverse=True)

        return {
            "error_profile": error_profile,
            "worst_samples": worst_samples,
            "strongest_residual_correlations": correlations[:max(0, int(max_correlations))],
        }

    @staticmethod
    def correlation_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return finite-sample Pearson and Spearman coefficients.

        Undefined correlations, including constant inputs, are returned as NaN.
        Warnings emitted for those mathematically valid edge cases are suppressed.
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            return float("nan"), float("nan")
        x, y = x[finite], y[finite]
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = float(np.corrcoef(x, y)[0, 1])
            spearman = float(stats.spearmanr(x, y).statistic)
        return pearson, spearman

def is_numeric_array(data: List[Any]) -> bool:
    arr = np.asarray(data)
    return np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
