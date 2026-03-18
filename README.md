# SR Agent

符号回归（Symbolic Regression）Agent，通过 LLM 调用工具分析数据并发现数学公式。

## 功能特点

- **数据探索**：自动计算数据的统计特征，帮助理解数据分布
- **公式评估**：使用 nd2py 符号引擎评估公式拟合能力，返回 MSE、R² 等指标
- **参数拟合**：支持 BFGS 算法自动优化公式中的参数
- **可扩展架构**：易于添加新的分析工具和 LLM 接口

## 项目结构

```
├── src/sr_agent/
│   ├── tools/          # 工具定义
│   │   ├── base_tool.py
│   │   ├── statistics.py
│   │   └── evaluate.py
│   ├── api/            # LLM 接口
│   ├── prompts/        # Prompt 模板
│   └── formulas/       # 公式处理
├── tests/              # 单元测试
├── _deps/              # 第三方依赖源码
└── playground/         # 实验性代码
```

## 安装

1. 创建虚拟环境：
```bash
conda create -p ./venv python=3.12 -y
conda activate ./venv  # Unix/Linux/MacOS
./venv/Scripts/activate  # Windows
```

2. 安装依赖：
```bash
# 激活环境后安装本项目及依赖
pip install -e ".[dev]"
```

3. 安装第三方依赖（目前只有 nd2py）：
```bash
git clone git@github.com:yuzhTHU/nd2py.git src/_deps/nd2py
pip install -e src/_deps/nd2py
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## 许可证

MIT License
