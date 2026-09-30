这个目录只用于存放被 `sr_agent` 私有化的第三方源码。

只有同时满足以下条件的代码才应该放在这里：

- 这个库没有合理的包结构，不能通过常规方式安装。
- 我们不打算把它作为独立项目维护或发布。
- 它只作为 `sr_agent` 的内部实现细节使用。

放入这里的依赖应使用 `sr_agent._vendor` 作为唯一导入入口，例如：

```python
from sr_agent._vendor.bad_package.xxx import yyy
```

不要让同一份源码同时支持 `import bad_package` 和 `import sr_agent._vendor.bad_package` 两种入口，否则 Python 会把它们加载成两套模块对象，可能导致 `isinstance`、类身份比较、注册表和缓存失效。

对于可以安装的第三方库，优先使用正常依赖声明或 `pip install`。如果需要本地调试源码，应将它们放在 `third-party/` 下，并对具体项目执行 `pip install -e ./third-party/<project>`。
