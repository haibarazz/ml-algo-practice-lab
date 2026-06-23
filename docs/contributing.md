# 贡献指南

这个仓库当前按个人学习项目维护。新增或修改模块时，优先保持结构一致，避免把一次性笔记混入练习模块。

## 新增模块

1. 复制 `templates/handwrite-module/` 的结构。
2. 在对应分组 `README.md` 和 `MODULE_INDEX.md` 中加入入口。
3. 确保模块包含主 Markdown、Notebook、`tests.py`、`solution.md` 和 `notes.md`。
4. 运行模块测试和全局检查。

## 模块要求

- 题面要声明手写实现约束。
- 带提示练习区只给实现台阶，不直接泄露答案。
- 无提示练习区要保留干净的函数签名。
- 测试脚本应能用 `python3 tests.py` 直接运行。
- 参考答案要解释关键实现原因。

## 提交前检查

```bash
python3 scripts/sync_docs.py
python3 scripts/check_docs_links.py
python3 scripts/test_all_modules.py
cd docs && npm run docs:build
```
