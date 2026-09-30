这个目录用于存放仍然保持独立项目身份的第三方源码。例如：
- 依赖库可以安装，但还在开发或调试阶段，需要本地 editable install。
- 依赖库是外部项目，我们希望保留源码供阅读、调试或打补丁。

本目录中的项目需要通过 editable install 的方式进行安装，例如：
```bash
pip install -e ./third-party/nd2py
```
直接通过 `import package_name` 导入这些第三方源码。不要把整个 `third-party/` 目录加入 `PYTHONPATH`，也不要通过 `import third-party.<project>` 导入其中的代码。

对于不能直接安装、或者需要调整源代码以适应 `sr_harness` 的第三方库，应将它放到 `src/sr_harness/` 中。
