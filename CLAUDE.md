# SR Agent 开发指南

## 运行环境

- 使用 conda prefix 环境 `./venv` 运行所有 Python 代码和测试
  - Windows 系统使用 `./venv/Scripts/python` 或 `./venv/python.exe`
  - Unix/Linux/MacOS 使用 `./venv/bin/python`

## 开发流程

1. 在 `src/sr_agent/tools/` 中实现新工具
2. 继承 `BaseTool` 并实现 `execute` 方法
3. 撰写详细的 docstring 作为 LLM 使用说明
4. 在 `tests/tools/` 中编写单元测试
5. 运行 `python -m pytest tests/ -v` 验证

## 代码规范

- 使用类型提示
- 使用 Google 风格的 docstring
- 项目代码放在 `src/sr_agent/` 目录下，测试代码放在 `tests/` 目录下
- 其它依赖代码放在 `src/_deps/` 目录下，通过 `pip install -e src/_deps/xxx` 安装或者在 `src/sr_agent/tools/` 中通过合适的方式包装使用

## 项目结构

```
src/sr_agent/
├── tools/         # 工具定义
├── api/           # LLM 接口
├── prompts/       # Prompt 模板
└── formulas/      # 公式处理
tests/
├── tools/         # 工具测试
└── conftest.py    # pytest 配置
```
