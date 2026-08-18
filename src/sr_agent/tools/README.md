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
- `self.context["evaluation_data"]`: 可选的隐藏验证集 `{变量名: np.ndarray}` 字典。工具只应在 `data`（训练集）上拟合；`evaluate()` 会自动在训练集和验证集上重新求值。

## 公式处理

在 `execute()` 中处理公式时，应尽量使用 `nd2py` 库（如 `nd.parse()`、`nd.BFGSFit()` 等）而非手动解析或计算。
- `nd2py` 是本项目中公式表示和求值的标准方式，与框架的其他部分（如 `evaluate()`、parser 等）紧密集成。

## 公式与指标约定

如果工具会产生公式（例如拟合、评估、变换、搜索），应直接使用 `self.evaluate()` 返回统一候选字典：
- `formula` 是一个字符串，表示工具产生的公式。
- `metrics` 包含该公式的评估指标。
- `is_candidate` 表示是否可参与 best formula 排名。
- `diagnostics` 包含可选的残差诊断；关闭时为空字典。

示例如下：
```python
return self.evaluate(
    f=formula_symbol,
    y=target_symbol,
)
```

`BaseTool.evaluate()` 会在目标恰为 `self.context["target"]` 且公式不包含目标变量时自动设置 `is_candidate=True`。
特殊工具可在获得结果后覆盖字段，例如公式展示文本不是 `f.to_str()` 时：

```python
result = self.evaluate(
    f=formula_symbol,
    y=target_symbol,
)
result["formula"] = custom_formula_text
return result
```

`f` 和 `y` 必须是 `nd2py.Symbol`。`evaluate()` 会分别在训练集和验证集上调用
`f.eval(data)` 与 `y.eval(data)`（不存在的验证集不会返回）。训练集和验证集结果统一保存在
`data_split_results["train"]` 与 `data_split_results["validation"]` 中，各自包含 `metrics`，以及可选的
`diagnostics`；不存在验证集时不返回 `validation` 字段。
代码模型如果已经在沙箱中得到预测数组，应调用 `BaseTool.calculate_metrics()` 复用相同的指标定义。
复杂度固定为 `len(f)`，不能由调用方覆盖。
残差诊断默认开启；内部候选筛选可显式传入 `show_diagnostics=False`，最终入选公式应保留诊断。

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

## TODO：统一评估数据划分模式

需要评估拟合结果的工具后续应共享一个统一接口，但本轮暂不实现，以避免同时改变所有工具的契约。计划支持：

- `train_equals_test`：在全部可见样本上拟合并评估，保持当前行为和向后兼容。
- `k_fold`：默认 5 折；每折只用训练折拟合，在测试折评价，最后在全量数据上重拟合用于输出公式。
- `out_of_domain`：按照指定变量、表达式或预定义 domain 标签切分；训练区间不得包含测试 domain。

建议实现为独立的共享组件，而不是让每个工具重复写切分代码：

```python
EvaluationConfig(
    mode="train_equals_test" | "k_fold" | "out_of_domain",
    n_splits=5,
    shuffle=True,
    random_seed=0,
    domain_expression=None,
    domain_test_range=None,
)
```

实施时应完成以下工作：

1. 在 `BaseTool` 附近增加只负责生成索引的 splitter；它不得接触或泄漏隐藏 benchmark test set。
2. 为拟合工具定义 `fit(train_indices)` 与 `predict(test_indices)` 的内部协议，并统一聚合每折指标、均值、标准差和最差折。
3. 明确区分 `selection_metrics`、交叉验证指标和全量重拟合后的 `refit_metrics`，避免用测试折选择常数后再次报告同一测试折。
4. OOD 模式要求显式提供 domain 变量/表达式及边界；支持低端、高端和区间外测试，并汇报训练、测试范围与有限覆盖率。
5. 保持 `is_candidate` 只对应最终全量重拟合公式；交叉验证中的临时公式不进入 top-k。
6. 先迁移 `polynomial_fit`、`power_law_fit`、`rational_fit`、`constant_fit`，再迁移 SINDy/PySR；为旧调用保留默认模式。

注意：这里的 “test” 是从 Agent 可见训练数据内部划分出的验证折，不得读取基准测试集或 OOD 隐藏答案。统一接口落地前，各工具现有的内部 holdout 参数仍保持局部实现。
