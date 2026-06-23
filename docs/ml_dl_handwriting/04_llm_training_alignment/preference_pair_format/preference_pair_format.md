# Preference Pair Format

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/04_llm_training_alignment/preference_pair_format/preference_pair_format.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 list / dict；不调用数据集框架。

## 原理最小说明

DPO/RM 常用偏好对：同一个 prompt 下有 chosen 和 rejected 两个回答。标准化数据结构能减少后续 collator 和 loss 的混乱。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python


def preference_pair_format(prompts, chosen, rejected):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python


def preference_pair_format(prompts, chosen, rejected):
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
def test_preference_pair_format():
    pairs = preference_pair_format(["Q1", "Q2"], ["good1", "good2"], ["bad1", "bad2"])
    assert pairs[0] == {"prompt": "Q1", "chosen": "good1", "rejected": "bad1"}
    try:
        preference_pair_format(["Q"], ["A"], [])
        assert False
    except ValueError:
        pass

test_preference_pair_format()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def preference_pair_format(prompts, chosen, rejected):
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, rejected must have the same length")
    pairs = []
    for p, c, r in zip(prompts, chosen, rejected):
        pairs.append({"prompt": p, "chosen": c, "rejected": r})
    return pairs
```

### 解析

1. 三个列表长度必须一致。
2. 每条样本保留 prompt/chosen/rejected 三个字段。
3. 这是后续 DPO/RM 的最小数据单元。


:::

## 工程要点 / 面试追问

### 核心公式

- 偏好样本通常包含同一个 prompt 下的一对回答：`(prompt, chosen, rejected)`。
- pairwise 学习关注 $\log p(chosen|prompt)-\log p(rejected|prompt)$ 的相对差异。

### 易错点

- `chosen`/`rejected` 顺序写反，训练信号会完全相反。
- prompt 没保留，后续无法拼接输入或计算条件概率。
- chosen/rejected 使用不同 chat template，导致比较不公平。
- 长度差异过大时，sum logprob 和 mean logprob 的选择会影响偏好强度。

### 面试追问

- DPO 数据和 SFT 数据有什么区别？
- pairwise preference 数据如何构造和清洗？
- 为什么 chosen/rejected 必须共享同一个 prompt？
- 偏好数据中长度偏置会怎样影响训练？