# SFT Label Mask

> Status: complete

## 题源线索

- Topic: SFT label mask。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

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

## 工程要点 / 面试追问

见 `notes.md`。
