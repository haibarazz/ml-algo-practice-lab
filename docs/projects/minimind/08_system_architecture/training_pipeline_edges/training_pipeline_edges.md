# Training Pipeline Edges

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/08_system_architecture/training_pipeline_edges/training_pipeline_edges.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `README.md:82-94`
- `trainer/train_tokenizer.py:24-166`
- `trainer/train_pretrain.py:83-171`
- `trainer/train_full_sft.py:84-172`
- `trainer/train_dpo.py:131-227`
- `trainer/rollout_engine.py:23-92`
- `scripts/serve_openai_api.py:28-245`

## 手写实现约束

只用 Python list/tuple。

## 原理最小说明

系统架构不是文件清单，而是数据和权重如何流动：raw text 训练 tokenizer，tokenizer 参与 dataset，dataset 驱动不同训练阶段，最终权重进入推理与评估。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_training_pipeline_edges():
    """TODO guided implementation."""
    # TODO 1: 列出核心节点
    # TODO 2: 用边表达依赖关系
    # TODO 3: 覆盖 tokenizer/data/model/train/infer/eval
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_training_pipeline_edges():
    """TODO blank implementation."""
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

## STOP HERE

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

### Source Mapping

- `README.md:82-94`
- `trainer/train_tokenizer.py:24-166`
- `trainer/train_pretrain.py:83-171`
- `trainer/train_full_sft.py:84-172`
- `trainer/train_dpo.py:131-227`
- `trainer/rollout_engine.py:23-92`
- `scripts/serve_openai_api.py:28-245`

### 常见坑

- 系统图要能解释权重从哪里来、数据在哪里变形。
- RL/Agentic RL 可以作为 DPO 后的扩展训练分支。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？