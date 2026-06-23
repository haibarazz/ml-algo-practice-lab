<p align="center">
  <img src="docs/public/ml-algo-practice-lab-hero.png" alt="ML Algorithm Practice Lab banner" width="100%">
</p>

<h1 align="center">ML Algorithm Practice Lab</h1>

<p align="center">
  面向机器学习/深度学习手撕题与开源 LLM 项目拆解的测试驱动练习库。
</p>

<p align="center">
  <strong>简体中文</strong>
  ·
  <a href="README_EN.md">English</a>
</p>

<p align="center">
  <a href="https://haibarazz.github.io/ml-algo-practice-lab/">在线文档</a>
  ·
  <a href="ml_dl_handwriting/MODULE_INDEX.md">手撕模块索引</a>
  ·
  <a href="projects/minimind/MODULE_INDEX.md">MiniMind 拆解</a>
  ·
  <a href="docs/contributing.md">贡献指南</a>
</p>

<p align="center">
  <img alt="Modules" src="https://img.shields.io/badge/modules-62-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-62%20module%20tests-green">
  <img alt="Docs" src="https://img.shields.io/badge/docs-VitePress-646CFF">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

## 项目简介

ML Algorithm Practice Lab 是一个开源学习项目，目标是把机器学习、深度学习和 LLM 工程中的核心机制拆成可阅读、可手写、可运行、可复盘的练习模块。

项目参考 [datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) 的课程站组织方式，并在此基础上强化两类能力：

- **无提示独立手撕能力**：每个模块同时提供带提示练习区和无提示练习区。
- **开源项目拆解能力**：把真实 LLM 项目拆成可测试的学习模块，而不是停留在源码阅读笔记。

这个项目希望解决的问题很直接：不只是“看懂算法”，而是能把关键机制写出来、跑通测试、解释实现细节，并能定位常见错误。

## 内容范围

| 主线 | 内容 | 入口 |
| --- | --- | --- |
| 机器学习/深度学习手撕 | 数学基础、传统机器学习、深度学习基础、Attention/Transformer、LLM 训练与对齐 | [ml_dl_handwriting](ml_dl_handwriting/MODULE_INDEX.md) |
| MiniMind 项目拆解 | 数据处理、模型结构、预训练、SFT、偏好对齐、推理、评估、系统架构 | [projects/minimind](projects/minimind/MODULE_INDEX.md) |
| 在线文档站 | VitePress 站点、模块导航、Colab 入口、折叠答案区 | [在线文档](https://haibarazz.github.io/ml-algo-practice-lab/) |

## 项目特点

- **测试驱动学习**：每个模块都有可直接运行的 `tests.py`。
- **题面透明**：不依赖隐藏判题，样例和断言都可以阅读、复盘和修改。
- **先带提示，再无提示**：第一次学习看实现台阶，面试复盘时撤掉提示独立完成。
- **贴近真实工程**：MiniMind 这类开源项目会被拆成函数、数据处理、损失、推理组件和系统链路。
- **文档可发布**：源码目录中的 Markdown 会同步到 VitePress 文档站，适合公开阅读和分享。

## 模块结构

每个练习模块默认遵循同一套页面顺序：

1. 原理最小说明
2. 带提示练习区
3. 无提示练习区
4. 测试区
5. `STOP HERE`
6. 参考答案与解析
7. 工程要点 / 面试追问

典型模块文件结构：

```text
module_name/
├── module_name.md
├── module_name.ipynb
├── tests.py
├── solution.md
└── notes.md
```

## 快速开始

克隆仓库：

```bash
git clone https://github.com/haibarazz/ml-algo-practice-lab.git
cd ml-algo-practice-lab
```

运行单个模块：

```bash
cd ml_dl_handwriting/00_math_primitives/softmax_stable
python3 tests.py
```

运行全部模块测试：

```bash
python3 scripts/test_all_modules.py
```

本地构建文档站：

```bash
cd docs
npm install
npm run docs:build
```

启动本地文档服务：

```bash
npm run docs:dev
```

## 仓库结构

```text
.
├── ml_dl_handwriting/          # 机器学习/深度学习手撕模块
├── projects/
│   └── minimind/               # MiniMind 开源项目拆解
├── docs/                       # VitePress 文档站
├── scripts/                    # 文档同步、链接检查、模块测试脚本
├── source-research/            # 题源和主题研究记录
└── templates/                  # 模块写作模板
```

## 文档发布流程

模块 Markdown 以源码目录为主，发布时同步到 `docs/`。

```bash
python3 scripts/sync_docs.py
python3 scripts/check_docs_links.py
cd docs && npm run docs:build
```

同步脚本会自动补充公开阅读需要的内容：

- 为带 Notebook 的模块增加 Colab / ModelScope 入口
- 把参考答案区域折叠，避免网页阅读时直接透题
- 生成 VitePress 可用的模块页面路径

## 验证基线

当前推荐的验证命令：

```bash
python3 scripts/test_all_modules.py
python3 scripts/check_docs_links.py
cd docs && npm run docs:build
```

当前基线：

- 62 个模块级测试通过
- 生成后的文档链接检查通过
- VitePress 构建通过

## 贡献

新增模块时建议从 [templates/handwrite-module](templates/handwrite-module/README.md) 开始，补充模块题面、Notebook、测试、答案和笔记，并把入口加入对应索引。

提交前请运行验证命令。更完整的流程见 [贡献指南](docs/contributing.md)。

## 开源协议

本项目采用 MIT License，见 [LICENSE](LICENSE)。
