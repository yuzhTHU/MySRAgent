# 工具开发指南

本目录包含可供 `SRAgent` 调用的工具。新工具需继承 `BaseTool`，注册一个稳定的工具名，并实现 `execute()` 方法和可选的 `format_result_dict()` 类方法。

## 最小示例

```python
from typing import Any, Dict

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("my_tool")
class MyTool(BaseTool):
    metadata = ToolMetadata(name="my_tool")

    def execute(self, value: str, limit: int = 10) -> Dict[str, Any]:
        """简短描述工具的功能。

        Args:
            value: value 的描述。
            limit: limit 的描述。
        """
        return {"value": value, "limit": limit}
```

## 必要组件

- 继承 `BaseTool`。
- 添加 `@BaseTool.register("tool_name")` 使工具被发现。
- 设置 `metadata = ToolMetadata(name="tool_name")`。
- 实现 `execute()`。
  - 其参数由 Agent 生成，因此应当尽量简单、且可序列化。
  - `execute()` 的返回值必须是一个 `Dict[str, Any]`，且不宜过长（会浪费 token）
  - 如果返回的结果字典较大，建议同时重写 `format_result_dict(cls, result)` 类方法，将字典格式化为一段较短的文本供 Agent 阅读，从而节省 token。
- 设置 `metadata`
  - `metadata.name` 是工具的唯一标识符，必须与 `@BaseTool.register(...)` 中的名称一致。
  - `metadata.description` 和 `metadata.parameters` 可选，如果不设置，会自动从 `execute()` 的 docstring 和签名推断。
    - `metadata.description` 从 docstring 中 `Args:` 之前的部分提取。
    - `metadata.parameters` 从 `execute()` 的签名、类型提示、默认值和 `Args:` 描述推断。
    - 如果 Agent 调用工具的成功率过低，建议检查自动推断的 `description` 和 `parameters` 是否准确反映了工具的功能和参数要求。
    - 当自动推断不够精确时（例如需要更严格的 JSON Schema 约束、枚举说明、嵌套对象或特殊格式），可以手动设置 `ToolMetadata(description=..., parameters=...)`。手动指定的元数据不会被 `BaseTool` 覆盖。

## 运行时上下文

工具实例化时传入的上下文可通过 `self.context` 访问，用于存放数据、模型、缓存等不应放在参数列表中由 Agent 生成的值。目前包含以下字段：
- `self.context["data"]`: 原始数据 DataFrame，{变量名: np.ndarray} 的字典格式。
- `self.context["target"]`: 目标变量名称字符串。除目标变量外的其他变量都可以作为公式中的自变量。

## 公式处理

在 `execute()` 中处理公式时，应尽量使用 `nd2py` 库（如 `nd.parse()`、`nd.BFGSFit()` 等）而非手动解析或计算。
- `nd2py` 是本项目中公式表示和求值的标准方式，与框架的其他部分（如 `evaluate()`、parser 等）紧密集成。

## 公式与指标约定

如果工具会产生公式（例如拟合、评估、变换、搜索），返回的字典中应当包含 `formula` 和 `metrics` 字段，其中：
- `formula` 是一个字符串，表示工具产生的公式。
- `metrics` 是由 `self.evaluate()` 方法生成的字典，包含该公式的评估指标。

示例如下：
```python
return {
    "formula": formula_str,
    "metrics": self.evaluate(y_pred=y_pred, y_true=y_true),
}
```

如果工具产生的公式有资格参与 best formula 评比，还应返回 `is_candidate: bool` 字段。
- 该字段为 `True` 表示此公式的 `metrics` 会被 `update_topk()` 记录并参与排名。
- 一般而言，当公式的目标变量恰为 `self.context["target"]` 且公式中不包含目标变量本身时，`is_candidate` 应为 `True`：

```python
return {
    "formula": formula_str,
    "metrics": self.evaluate(y_pred=y_pred, y_true=y_true),
    "is_candidate": (y == self.context["target"]) and (y not in x_list),
}
```

## 错误处理

- **不影响正常运行的警告**：将警告信息汇总到结果字典的 `exceptions` 字段，以供 Agent 参考。
  - 例如某些输入变量解析失败但其余变量仍可用时，将失败信息追加到 `exceptions` 中即可。
- **导致无法正常运行的错误**：直接 `raise` 抛出异常即可。
  - 抛出时可将已积累的警告信息一并包含在错误消息中，方便 Agent 理解上下文。例如：
    ```python
    if some_error_condition:
        error_message = f"Error occurred due to XXX. Previous warnings: {exceptions}"
        raise RuntimeError(error_message)
    ```
  - `BaseTool.__call__()` 会接住工具 `execute()` 方法中抛出的异常，将其格式化为错误信息返回给 Agent。

## 工具注册

- `@BaseTool.register("tool_name")` 会使工具出现在 Agent 可用的工具列表中。
- 因此，**尚未实现完善或测试不充分的工具不应注册** —— Agent 调用这类工具的成功率太低，不注册即可确保 Agent 无法使用它，避免浪费调用次数和 token。
- 可以将 `@BaseTool.register(...)` 注释掉来暂时取消注册，待工具成熟后再启用。