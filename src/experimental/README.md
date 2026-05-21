# 实验性代码

本目录用于存放开发中、实验性、暂不稳定的代码。这些内容不属于正式的
`sr-agent` 包，也不会随 PyPI 分发。

本目录下的 package 可以依赖 `src/sr_agent/` 中已有的稳定代码，例如：

```python
from sr_agent import SRAgent
```

依赖方向应保持单向：`src/experimental/` 可以使用 `src/sr_agent/`，但 `src/sr_agent/` 不应该反向导入或依赖 `src/experimental/` 中的内容。

当某个实验性 package 足够成熟后，可以将稳定部分移动到 `src/sr_agent/` 中的对应位置，例如 `src/sr_agent/tools/`。

入口脚本放到 `scripts/` 或 `analysis/` 中，通过将 `src/experimental` 加入 `sys.path` 的方式直接导入其中的 package：

```python
import sys
from pathlib import Path
ROOT = Path("..").absolute()  # 仓库根目录
sys.path.append(str(ROOT / "src" / "experimental"))

import package_name
```

这些入口脚本产生的运行结果应保存到：

```text
logs/experimental/package_name/
```
