# 测试指南

## 运行测试

```bash
# 默认：只运行快速单元测试
./venv/python -m pytest tests/ -v

# 运行所有测试（包括集成测试和付费测试）
./venv/python -m pytest tests/ -v -m ""

# 只运行集成测试
./venv/python -m pytest tests/ -v -m integration

# 只运行付费测试
./venv/python -m pytest tests/ -v -m paid
```

## 测试分类与标记

所有测试通过 pytest marker 分为三类：

| 标记 | 含义 | 默认是否运行 |
|------|------|:---:|
| （无标记） | 快速单元测试，不依赖外部服务 | 是 |
| `@pytest.mark.slow` | 集成测试，较慢，可能依赖本地资源但不产生费用 | 否 |
| `@pytest.mark.paid` | 付费测试，会调用真实 LLM API 并产生费用 | 否 |

默认配置下，`slow` 和 `paid` 标记的测试被排除。这是通过 `pyproject.toml` 中的 `markers` 和 `addopts` 实现的。

```python
# 快速单元测试 —— 无需标记
def test_parse_single_action():
    parser = TextParser(tool_list=None)
    actions = parser.parse_response("Action: foo(x=1)")
    assert len(actions) == 1

# 集成测试
@pytest.mark.slow
def test_format_then_parse_roundtrip():
    ...

# 付费测试
@pytest.mark.paid
def test_llm_generates_parseable_action():
    ...
```

## 测试编写风格

**优先使用函数式测试**（独立函数 + 装饰器），而非类式测试：

```python
# 推荐 ✓
def test_infer_tool_description():
    assert UnitSampleTool.metadata.description == "Run a sample tool."

# 不推荐 ✗
class TestToolMetadata:
    def test_infer_tool_description(self):
        assert UnitSampleTool.metadata.description == "Run a sample tool."
```

使用 `@pytest.fixture` 共享测试前置条件：

```python
@pytest.fixture
def sample_data():
    return {"x1": np.array([1.0, 2.0, 3.0]), "y": np.array([2.0, 4.0, 6.0])}

def test_execute_analyzes_selected_variables(sample_data):
    tool = StatisticsTool(data=sample_data)
    result = tool.execute(variables=["x1"])
    assert "x1" in result["statistics"]
```

使用 `@pytest.mark.parametrize` 覆盖多种输入：

```python
@pytest.mark.parametrize("annotation, schema", [
    (str, {"type": "string"}),
    (int, {"type": "integer"}),
])
def test_parse_args_typehints(annotation, schema):
    assert BaseTool.parse_args_typehints(annotation) == schema

```

注：如果一组测试共享较多的 setup 逻辑且函数式写法显得冗余，也可以使用类式测试。


## 目录结构

```
tests/
├── conftest.py          # pytest 全局配置（sys.path 等）
├── test_xxx.py          # SRHarness 测试
├── tools/               # 工具测试
│   ├── test_xxx.py
├── parser/              # 解析器测试
│   ├── test_xxx.py
├── api/                 # API 测试
│   └── test_xxx.py
└── xxx/                 # 更多模块测试
```

## 编写新测试

1. **测试文件命名**：`test_<模块名>.py`，放在 `tests/` 下对应子目录中。
2. **为每个测试函数添加适当的标记**
   - 无标记：确保可以在无网络、无 API key 的环境下快速运行。
   - 集成测试标记 `@pytest.mark.slow`：用于端到端冒烟测试（如 parser 的 format→parse 往返测试）。
   - 付费测试标记 `@pytest.mark.paid`：用于调用真实 LLM API 的测试。这类测试应做好容错处理（如 API 不可用时 `pytest.skip()`），避免因网络或额度问题导致 CI 失败。
3. **避免依赖注册表状态**：如果测试中需要注册临时工具，使用唯一的名称前缀（如 `unit_`）避免与其他测试冲突。
