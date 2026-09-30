# SRAgent 框架说明

## SRAgent 整体架构

SRAgent 是一个基于 LLM 的符号回归 Agent。其核心是一个 **"请求 LLM → 解析工具调用 → 调用工具 → 格式化消息"** 的循环，循环的 pipeline 由四个参数控制：

```
for R in 1..max_restart_loop:       # best-solution restart 次数
  for C in 1..global_width:         # 独立对话分支数量
    for L in 1..max_refinement_depth:  # 每个分支的迭代轮数
      for K in 1..local_sample_size:   # 每轮 LLM 重复采样次数
        ... (单次迭代)
```

- **R (Restart)**：外层重启循环。每轮重启会用历史最优结果构建新的初始 prompt，引导 LLM 在已有成果上继续搜索。
  - 注：现在尚未实现基于历史最优结果更新初始 prompt 的功能，因此建议保持 R=1
- **C (Conversation branch)**：独立对话分支，每个分支从相同的初始 prompt 出发独立探索。
- **L (Refinement step)**：单个分支内的对话迭代，每轮包含一次完整的 prompt 构建、LLM 请求、工具调用、buffer 更新。
- **K (Local sample)**：单次 LLM 请求的重复采样次数，产生多个候选响应，产生最佳结果的响应会被追加到 buffer 中，其余结果中不涉及公式评估的工具调用结果也会被追加到 buffer 中以供后续参考。

## SRAgent 核心组件

SRAgent 持有两个核心组件：

`tool_list: List[BaseTool]`：可用的工具实例列表。每个工具接收调用参数，返回一个结果字典。
- 工具的详细开发指南见 [`tools/README.md`](tools/README.md)。
- 工具的调用流程如下：
  1. 初始化 SRAgent 时，工具被实例化并拿到运行时上下文（数据、目标变量等）。
  2. LLM 产生工具调用（名称 + 参数）
  3. Agent 根据名称在 `tool_list` 中查找对应工具实例
  4. 调用 `tool(**params)`，返回 `ToolCallResult`（包含 `ok`、`result`、`result_str`、`meta_data`）

`llm_api: LLMAPI`：LLM API 实例，负责与 LLM 服务交互。
- 调用方式如下：
  ```python
  llm_result = self.llm_api(messages, n=local_sample_size)
  ```
- `messages` 是对话消息列表，格式为 `[{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}, ...]`。
- `llm_result` 是一个 `LLMResult` 对象，可迭代获取 LLM 的响应：
  ```python
  for content, tool_calls, message in llm_result:
      # content: LLM 响应中的文字内容
      # tool_calls: 从响应中解析出的工具调用列表 (List[ToolCall])
      #   - 文本解析模式 (text/json): 从 content 中解析
      #   - OpenAI 模式: 从 message 的 tool_calls 字段直接提取
      # message: 可直接追加到 messages 中以延续对话的消息字典
  ```
- 迭代完成后，`llm_result` 中保留用量统计：
  ```python
  llm_result.usage   # {'token': {...}, 'price': {...}}
  ```

## SRAgent 单次迭代流程

每一轮 (L) 的执行步骤如下：

1. **构建 Prompt**：根据当前 buffer（对话历史）构建 messages。
2. **请求 LLM**：调用 `llm_api(messages)`，得到 K 个候选响应。
3. **执行工具**：解析每个响应中的 tool_calls，查找并调用对应工具，得到结果列表。
4. **更新 Buffer**：选择产生最优 mse 的分支，将其 message 和工具结果追加到 buffer；其他分支中不涉及公式评估的工具结果也会追加，避免丢失有用信息。
5. **更新 Top-K**：将 `is_candidate=True` 的结果按 mse 排序记录。
6. **日志输出**：打印当前最优结果、工具调用统计、token/费用用量等。

## 目录结构

```
src/sr_agent/
├── sr_agent.py      # SRAgent 主类，包含 fit() 主循环和各个步骤方法
├── api/             # LLM API 封装
│   ├── core.py      # LLMResult、ToolCall 等核心数据结构
│   ├── llm_api.py   # LLMAPI 基类和工厂方法
│   └── *_api.py     # 各提供商实现 (OpenAI, DeepSeek, Gemini, ...)
├── parser/          # 工具调用解析器
│   ├── base_parser.py   # BaseParser 基类
│   ├── *_parser.py   # 具体解析器实现 (TextParser, JSONParser, ...)
├── tools/           # 工具定义
│   ├── base_tool.py # BaseTool 基类
│   └── ...          # 具体工具实现
├── utils/           # 工具函数
├── prompts/         # Prompt 模板（暂时用不到）
├── buffer/          # 消息管理（暂时用不到）
└── skills/          # 技能文档（暂时用不到）
```
