本目录用于存放被 `sr_harness` 私有化的第三方源码。

具体而言，sr_harness 需要利用第三方的代码和/或工具，尽管绝大多数工具可以通过 `pip install` 等方式安装，某些代码可能：
1. 不存在合理的包结构、无法通过常规方式安装；
2. 需要经过适当修改或包装以适应 SRHarness 的调用；
我们将这类代码作为 `sr_harness` 的内部实现细节，安装到这一目录中。 

这一目录中的依赖应使用 `sr_harness._vendor` 作为唯一导入入口，例如：
```python
import sr_harness._vendor.vendor_package
```
对于可以安装的第三方库，优先使用正常依赖声明和/或 `pip install`，不要将它放入这一目录。
不要让同一份源码同时支持 `import vendor_package` 和 `import sr_harness._vendor.vendor_package` 两种入口，否则 Python 会把它们加载成两套模块对象，可能导致 `isinstance`、类身份比较、注册表和缓存失效。
