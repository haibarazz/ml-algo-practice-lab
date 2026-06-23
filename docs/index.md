# ML Algorithm Practice Lab

![ML Algorithm Practice Lab](/ml-algo-practice-lab-hero.png)

ML Algorithm Practice Lab 是一个开源学习项目，用于把机器学习、深度学习和 LLM 工程中的核心机制拆成可阅读、可手写、可运行、可复盘的练习模块。

项目参考 Datawhale `llm-algo-leetcode` 的课程站组织方式，并扩展出两条更明确的能力主线：无提示独立手撕能力，以及从真实开源 LLM 项目中抽取可练习模块的能力。

## 学习主线

| 主线 | 内容 | 入口 |
| --- | --- | --- |
| 机器学习/深度学习手撕 | 数学基础、传统机器学习、深度学习基础、Attention/Transformer、LLM 训练与对齐 | [模块索引](./ml_dl_handwriting/MODULE_INDEX.md) |
| 开源项目拆解 | 以 MiniMind 为样例，把真实 LLM 项目拆成数据、模型、训练、推理、对齐、评估和系统架构模块 | [MiniMind 索引](./projects/minimind/MODULE_INDEX.md) |

## 模块结构

每个手撕模块默认包含：

1. 原理最小说明
2. 带提示练习区
3. 无提示练习区
4. 测试区
5. STOP HERE
6. 参考答案与解析
7. 工程要点 / 面试追问

## 快速开始

在线阅读时，先从左侧导航选择主线和模块。需要本地练习时，克隆仓库后进入对应模块目录，运行该模块的 `tests.py`。

```bash
python3 tests.py
```

更完整的环境、测试和发布说明见 [使用指南](./guide.md) 与 [维护手册](./maintenance.md)。
