# SFT Assistant Label Mask

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/01_tokenizer_and_data/sft_assistant_label_mask/sft_assistant_label_mask.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `dataset/lm_dataset.py:58-119`

## 手写实现约束

只用 Python list；用整数 token 模拟 chat template。

## 原理最小说明

SFT 不是让模型学习 user/system 原文，而是给定上下文后学习 assistant 回复。MiniMind 通过寻找 assistant 起始 token 序列和 eos token 序列，把其它位置 label 置为 `-100`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
    """TODO guided implementation."""
    # TODO 1: 初始化全 -100 labels
    # TODO 2: 扫描 assistant_bos 子序列
    # TODO 3: 找到后续 assistant_eos 子序列
    # TODO 4: 只复制回复 span 到 labels
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
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
def test_generate_assistant_labels():
    ids = [9, 101, 102, 5, 6, 201, 7, 101, 102, 8, 201, 0]
    labels = generate_assistant_labels(ids, assistant_bos=[101, 102], assistant_eos=[201], max_length=len(ids))
    assert labels == [-100, -100, -100, 5, 6, 201, -100, -100, -100, 8, 201, -100]


test_generate_assistant_labels()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(assistant_bos)] == assistant_bos:
            start = i + len(assistant_bos)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(assistant_eos)] == assistant_eos:
                    break
                end += 1
            stop = min(end + len(assistant_eos), max_length)
            for j in range(start, stop):
                labels[j] = input_ids[j]
            i = stop
        else:
            i += 1
    return labels
```

### 解析

1. 真实项目中的 assistant_bos/eos 来自 tokenizer 编码。
2. 如果 eos 被截断，labels 会延伸到 max_length。


:::

## 工程要点 / 面试追问

### Source Mapping

- `dataset/lm_dataset.py:58-119`

### 常见坑

- 真实项目中的 assistant_bos/eos 来自 tokenizer 编码。
- 如果 eos 被截断，labels 会延伸到 max_length。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？