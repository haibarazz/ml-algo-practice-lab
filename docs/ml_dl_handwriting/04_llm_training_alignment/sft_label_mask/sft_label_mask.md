# SFT Label Mask

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/04_llm_training_alignment/sft_label_mask/sft_label_mask.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 list / NumPy；不调用数据处理框架。

## 原理最小说明

SFT 常只训练 assistant answer 部分。prompt token 和 padding token 的 label 通常设为 `-100`，让 loss 忽略这些位置。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def sft_label_mask(input_ids, prompt_lengths, pad_token_id=0, ignore_index=-100):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np

def sft_label_mask(input_ids, prompt_lengths, pad_token_id=0, ignore_index=-100):
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
def test_sft_label_mask():
    ids = np.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    labels = sft_label_mask(ids, [2, 1], pad_token_id=0)
    assert labels.tolist() == [[-100, -100, 3, 4, -100], [-100, 6, 7, -100, -100]]

test_sft_label_mask()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def sft_label_mask(input_ids, prompt_lengths, pad_token_id=0, ignore_index=-100):
    input_ids = np.asarray(input_ids, dtype=np.int64)
    labels = input_ids.copy()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = ignore_index
    labels[input_ids == pad_token_id] = ignore_index
    return labels
```

### 解析

1. labels 初始复制 input_ids。
2. 每条样本按 prompt length 屏蔽 prompt。
3. padding 统一屏蔽。
4. 训练时 CE 会忽略 -100。


:::

## 工程要点 / 面试追问

### 核心公式

- $L=-\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}\log p_\theta(y_t|y_{<t},prompt)$。
- prompt/pad 位置 label 通常设为 `ignore_index`，只让 answer token 贡献 loss。

### 易错点

- 把 prompt 也计入 loss，模型会被训练去复述用户输入。
- pad 没 mask，batch padding 会污染 loss。
- `prompt_lengths` 和 batch 样本对不齐。
- 多轮对话中 assistant/user/system 边界没有明确标注。

### 面试追问

::: details 参考回答：SFT 为什么通常只训练 answer token？

SFT 训练的是在 prompt 条件下生成 answer 的能力，prompt 是条件，不是要模型学习输出的目标。只训练 answer token 可以避免模型被奖励去复述用户问题或 system 指令。

:::

::: details 参考回答：多轮对话中 label mask 应如何设计？

多轮对话中通常只让 assistant 回复位置参与 loss，system/user token 和 padding 都设为 ignore。若要训练特定轮次，也要严格根据 chat template 标记每段角色边界。

:::

::: details 参考回答：如果把 prompt token 也计入 loss，会带来什么问题？

prompt token 也计入 loss 会让模型优化“预测用户输入”的概率，而这不是部署时的生成任务。它还会让长 prompt 样本贡献过多 loss，稀释 answer 部分的监督信号。

:::

::: details 参考回答：不同 chat template 会怎样影响 mask 边界？

chat template 决定 system/user/assistant 的特殊 token、分隔符和起止位置。模板不同会改变 answer 从哪里开始、哪些 token 属于角色标记，因此 label mask 边界必须和模板一致。

:::