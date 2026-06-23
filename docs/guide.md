# 使用指南

本项目同时服务两个场景：在线阅读和本地手写练习。

## 在线阅读

网站中的模块页来自源码目录下的 Markdown 文件，并由 `scripts/sync_docs.py` 同步生成到 `docs/`。在线页面适合快速浏览题面、理解原理、查看测试方式和折叠后的参考答案。

## 本地练习

推荐从仓库根目录进入具体模块目录：

```bash
cd ml_dl_handwriting/00_math_primitives/softmax_stable
python3 tests.py
```

每个模块通常包含：

- `{module}.md`: 阅读版题面
- `{module}.ipynb`: Notebook 练习界面
- `tests.py`: 本地验证脚本
- `solution.md`: 参考答案
- `notes.md`: 易错点和面试追问

## 环境边界

- 第一阶段手撕题以 Python、NumPy 和少量 PyTorch 为主。
- `projects/minimind/` 的模块只保留学习拆解和测试，不提交真实源码目录。
- `projects/minimind/source/` 是本地参考源码目录，已被 `.gitignore` 忽略。

## 常用命令

```bash
# 生成网站 Markdown 镜像
python3 scripts/sync_docs.py

# 跑全部模块测试
python3 scripts/test_all_modules.py

# 检查站内链接
python3 scripts/check_docs_links.py

# 构建网站
cd docs
npm run docs:build
```
