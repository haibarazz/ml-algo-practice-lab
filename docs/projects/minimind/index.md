# MiniMind Project Dissection

这个文件夹把 MiniMind 这类开源 LLM 项目拆成可学习、可手写、可测试的模块。源码本地 clone 在 `source/`，但该目录被本仓库 `.gitignore` 忽略；我们提交的是学习拆解、练习和测试。

## How to Use

1. 先读 `SOURCE_MAP.md`，知道真实源码入口在哪里。
2. 再按 `MODULE_INDEX.md` 的顺序做练习。
3. 每个模块遵循同一结构：原理最小说明 -> 带提示练习区 -> 无提示练习区 -> 测试区 -> STOP HERE -> 参考答案与解析。
4. 做完模块后回到 `source/` 里的真实源码，对照 `Source Mapping` 的路径和行号。

## Sections

- `00_project_map`: 项目目录、核心入口、训练/推理路径。
- `01_tokenizer_and_data`: tokenizer、数据格式、dataset、label mask。
- `02_model_architecture`: RMSNorm、RoPE、GQA、SwiGLU、MoE。
- `03_pretraining`: causal LM loss、学习率、梯度累积。
- `04_sft`: SFT 数据模板和 thinking 标签处理。
- `05_preference_alignment`: DPO、per-token logprob、rollout 相关核心量。
- `06_inference`: generation、KV cache、采样过滤。
- `07_evaluation`: perplexity 等评估基本量。
- `08_system_architecture`: 从数据到训练、推理和评估的系统级架构。

## Dissection Rule

MiniMind 拆解按学习路径组织，而不是严格按原项目目录组织。每个模块必须能落到函数、类、损失、数据处理步骤或推理组件的 TODO 实现。
