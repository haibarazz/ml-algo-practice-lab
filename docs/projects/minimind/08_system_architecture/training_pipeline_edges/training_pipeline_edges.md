# MiniMind 系统依赖图：数据流、权重流和服务流

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/08_system_architecture/training_pipeline_edges/training_pipeline_edges.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


把 MiniMind 的源码节点串成可执行的系统架构图，理解训练产物如何进入推理服务。

## 学习目标

- 理解数据、模型、训练脚本、checkpoint、推理服务之间的依赖。
- 区分训练流、对齐流、rollout 流和服务流。
- 掌握从源码中抽取系统级边的能力。

## MiniMind 源码定位

- `dataset/lm_dataset.py`
- `model/model_minimind.py`
- `trainer/*.py`
- `scripts/*.py`
- `eval_llm.py`

## 源码机制详解

训练流从 dataset 开始：不同 Dataset 产出不同 batch 结构，预训练/SFT 是 `(input_ids, labels)`，DPO 是 chosen/rejected 的 x/y/mask，RL/Agent 则先产出 prompt/messages/tools，再 rollout。
模型流集中在 `MiniMindForCausalLM`：训练时返回 loss/logits/aux_loss，推理时通过 generate 循环返回 generated_ids 和可选 past_kv。MoE 模型还会在训练目标中额外加入 aux_loss。
权重流由 `trainer_utils.lm_checkpoint` 和各训练脚本共同管理：训练脚本保存 `.pth`，可从 `from_weight` 加载已有权重继续 SFT/DPO/LoRA。服务流再通过 `eval_llm.py`、`serve_openai_api.py`、`web_demo.py` 读取模型并暴露交互入口。

## 关键公式与数据流

- 预训练边：`PretrainDataset -> MiniMindForCausalLM(loss) -> AdamW -> checkpoint`。
- SFT 边：`SFTDataset(label mask) -> same causal LM loss -> full_sft weight`。
- DPO 边：`DPODataset -> policy/ref logprob -> DPO loss -> dpo weight`。
- 推理边：`tokenizer prompt -> generate -> KV cache -> sampled tokens -> decode`。

## 为什么练这个

- 这个练习训练你从工程代码中抽取系统图，而不是只读单个函数。
- 系统图能帮助读者判断新源码机制应该挂在哪条主线下。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_training_pipeline_edges():
    """带提示实现。"""
    # TODO 1: 列出核心节点
    # TODO 2: 用边表达依赖关系
    # TODO 3: 覆盖 tokenizer/data/model/train/infer/eval
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_training_pipeline_edges():
    """无提示实现。"""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现无提示函数后直接运行测试区代码。

```python
def test_build_training_pipeline_edges():
    edges = build_training_pipeline_edges()
    assert ("raw_text", "tokenizer") in edges
    assert ("tokenizer", "pretrain_dataset") in edges
    assert ("pretrained_weights", "sft_training") in edges
    assert ("aligned_weights", "openai_api_server") in edges


test_build_training_pipeline_edges()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def build_training_pipeline_edges():
    return [
        ("raw_text", "tokenizer"),
        ("tokenizer", "pretrain_dataset"),
        ("tokenizer", "sft_dataset"),
        ("model_config", "model_architecture"),
        ("pretrain_dataset", "pretrain_training"),
        ("model_architecture", "pretrain_training"),
        ("pretrain_training", "pretrained_weights"),
        ("pretrained_weights", "sft_training"),
        ("sft_dataset", "sft_training"),
        ("sft_training", "sft_weights"),
        ("sft_weights", "dpo_training"),
        ("preference_pairs", "dpo_training"),
        ("dpo_training", "aligned_weights"),
        ("aligned_weights", "openai_api_server"),
        ("aligned_weights", "evaluation"),
    ]
```

### 解析

1. 系统图要能解释权重从哪里来、数据在哪里变形。
2. RL/Agentic RL 可以作为 DPO 后的扩展训练分支。


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- 预训练边：`PretrainDataset -> MiniMindForCausalLM(loss) -> AdamW -> checkpoint`。
- SFT 边：`SFTDataset(label mask) -> same causal LM loss -> full_sft weight`。
- DPO 边：`DPODataset -> policy/ref logprob -> DPO loss -> dpo weight`。
- 推理边：`tokenizer prompt -> generate -> KV cache -> sampled tokens -> decode`。

### 易错点

- 把所有 trainer 都看成同一个训练脚本，会忽略 batch 结构和 loss 差异。
- 只画数据流不画权重流，会看不懂 `from_weight` 和阶段继承。
- 只画训练不画服务，会漏掉 tokenizer/template 在推理中的一致性问题。

### 面试追问

::: details 参考回答：MiniMind 的系统图里为什么要区分数据流和权重流？

数据流解释一个 batch 如何进入 loss，权重流解释不同训练阶段如何继承和保存模型。SFT、DPO、LoRA 的关键差别很多时候不在 forward，而在权重从哪里来、保存到哪里去。

:::

::: details 参考回答：为什么服务流也应该放进学习地图？

LLM 项目的目标不是只训练出 loss，而是能推理、对话或提供 API。服务流会暴露 tokenizer、chat template、KV cache、采样参数等与训练同样关键的工程约束。

:::