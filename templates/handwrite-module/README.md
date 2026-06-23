# Handwrite Module Template

复制这个模板目录来创建新的手撕模块。

## Files

- `{module_name}.ipynb`: 主练习 Notebook，包含带提示区、无提示区、测试区和答案区。
- `{module_name}.md`: Markdown 阅读版。
- `solution.md`: 参考答案与解析。
- `tests.py`: 可运行测试。
- `notes.md`: 公式、易错点和面试追问。

## Writing Rules

- 先写带提示区，帮助第一次学习；再写无提示区，用于撤掉提示后的自测。
- 带提示区只拆台阶，不直接泄露答案。
- `solution.md` 必须解释为什么这样实现，而不是只贴代码。
- `tests.py` 采用轻量脚本风格：固定样例、直接 `assert`、可用 `python tests.py` 运行，不引入复杂测试框架。
- 测试只覆盖核心正确性和一两个常见错误，不追求完整工程级测试。
- 每个模块都要说明它在真实 ML/LLM 系统中的位置。
- 每个模块必须声明手写实现约束，例如只允许 `list` / NumPy，不允许 sklearn、torch 内置层或现成指标函数。

## Test Style

参考 `llm-algo-leetcode` 的验证方式：题目区应该失败，答案区应该通过。当前项目的模块可以先不使用 Notebook 提取脚本，而是在每个模块内维护一个清晰的 `tests.py`。

本项目不是 ACM / OJ 风格：

- 不以 stdin/stdout 作为主要交互方式。
- 不追求隐藏测试点和在线判题。
- 每个练习围绕一个函数、类或训练组件展开。
- 题面给出函数签名、张量/数组 shape、TODO 位置和测试函数。
- 学习路径遵循“原理最小说明 -> 带提示练习区 -> 无提示练习区 -> 测试区 -> STOP HERE -> 参考答案与解析”。

标准 section 顺序：

1. 原理最小说明
2. 带提示练习区
3. 无提示练习区
4. 测试区
5. `STOP HERE`
6. 参考答案与解析
