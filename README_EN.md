<p align="center">
  <img src="docs/public/ml-algo-practice-lab-hero.png" alt="ML Algorithm Practice Lab banner" width="100%">
</p>

<h1 align="center">ML Algorithm Practice Lab</h1>

<p align="center">
  A test-driven practice lab for ML/DL handwriting problems and open-source LLM project dissection.
</p>

<p align="center">
  <a href="README.md">简体中文</a>
  ·
  <strong>English</strong>
</p>

<p align="center">
  <a href="https://haibarazz.github.io/ml-algo-practice-lab/">Documentation</a>
  ·
  <a href="ml_dl_handwriting/MODULE_INDEX.md">Module Index</a>
  ·
  <a href="projects/minimind/MODULE_INDEX.md">MiniMind Track</a>
  ·
  <a href="docs/contributing.md">Contributing</a>
</p>

<p align="center">
  <img alt="Modules" src="https://img.shields.io/badge/modules-62-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-62%20module%20tests-green">
  <img alt="Docs" src="https://img.shields.io/badge/docs-VitePress-646CFF">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

## Overview

ML Algorithm Practice Lab is an open-source learning repository for turning ML and LLM engineering concepts into small, runnable, reviewable exercises.

The project is inspired by [datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode), and extends that style in two directions:

- **Independent handwriting practice**: every module contains both guided and unguided exercise sections.
- **Open-source project dissection**: selected real LLM projects are decomposed into testable learning modules instead of only being summarized as reading notes.

The goal is simple: move from "I understand the idea" to "I can implement, test, explain, and debug the core mechanism".

## What Is Included

| Track | Scope | Entry |
| --- | --- | --- |
| ML/DL Handwriting | Math primitives, classical ML, deep learning basics, Attention/Transformer, LLM training and alignment | [ml_dl_handwriting](ml_dl_handwriting/MODULE_INDEX.md) |
| MiniMind Dissection | Data processing, model architecture, pretraining, SFT, preference alignment, inference, evaluation, system architecture | [projects/minimind](projects/minimind/MODULE_INDEX.md) |
| Documentation Site | VitePress site with generated module pages, sidebar navigation, Colab links, and folded answer sections | [Online docs](https://haibarazz.github.io/ml-algo-practice-lab/) |

## Design Principles

- **Test-driven learning**: each module has a local `tests.py` that can be run directly.
- **No hidden judge dependency**: examples and assertions are visible so learners can inspect the expected behavior.
- **Guided first, blank second**: the same concept is practiced with hints first, then without hints.
- **Source-aware dissection**: project modules keep explicit links back to the real implementation ideas they are derived from.
- **Docs as a published surface**: Markdown content is mirrored into a VitePress site for public reading.

## Module Format

Each practice module follows a consistent page order:

1. Minimal concept explanation
2. Guided exercise
3. Unguided exercise
4. Test section
5. `STOP HERE`
6. Reference solution and explanation
7. Engineering notes and interview follow-ups

Typical module files:

```text
module_name/
├── module_name.md
├── module_name.ipynb
├── tests.py
├── solution.md
└── notes.md
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/haibarazz/ml-algo-practice-lab.git
cd ml-algo-practice-lab
```

Run one module:

```bash
cd ml_dl_handwriting/00_math_primitives/softmax_stable
python3 tests.py
```

Run all module tests:

```bash
python3 scripts/test_all_modules.py
```

Build the documentation site locally:

```bash
cd docs
npm install
npm run docs:build
```

Start the local docs server:

```bash
npm run docs:dev
```

## Repository Structure

```text
.
├── ml_dl_handwriting/          # ML/DL handwriting modules
├── projects/
│   └── minimind/               # MiniMind open-source project dissection
├── docs/                       # VitePress documentation site
├── scripts/                    # docs sync, link check, module test runners
├── source-research/            # topic/source research notes
└── templates/                  # module authoring template
```

## Documentation Workflow

Module Markdown files are maintained in the source directories. The published documentation mirrors them into `docs/`.

```bash
python3 scripts/sync_docs.py
python3 scripts/check_docs_links.py
cd docs && npm run docs:build
```

The sync step also adds public-reading conveniences:

- Colab and ModelScope entry blocks for modules with notebooks
- folded reference-answer sections for web readers
- VitePress-compatible navigation paths

## Verification

Current verification commands:

```bash
python3 scripts/test_all_modules.py
python3 scripts/check_docs_links.py
cd docs && npm run docs:build
```

Expected baseline:

- 62 module-level tests pass
- all generated documentation links resolve
- VitePress build completes

## Contributing

Contributions should keep the module format consistent and runnable. For new modules, start from [templates/handwrite-module](templates/handwrite-module/README.md), add the module to the relevant index, and run the verification commands before opening a pull request.

See [docs/contributing.md](docs/contributing.md) for the detailed workflow.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
