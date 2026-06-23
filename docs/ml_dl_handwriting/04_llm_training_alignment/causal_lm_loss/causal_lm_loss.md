# Causal LM Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/04_llm_training_alignment/causal_lm_loss/causal_lm_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: Causal LM next-token loss。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Causal LM 用当前位置 logits 预测下一个 token：

- logits 使用 `[:, :-1, :]`
- labels 使用 `[:, 1:]`

再对有效 label 计算交叉熵。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def causal_lm_loss(logits, labels, ignore_index=-100):
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

def causal_lm_loss(logits, labels, ignore_index=-100):
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
def test_causal_lm_loss():
    logits = np.array([[[0, 5, 0], [0, 0, 5], [5, 0, 0]]], dtype=float)
    labels = np.array([[0, 1, 2]])
    loss = causal_lm_loss(logits, labels)
    assert loss < 0.02
    labels_masked = np.array([[0, -100, 2]])
    assert np.isfinite(causal_lm_loss(logits, labels_masked))

test_causal_lm_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def causal_lm_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    mask = flat_labels != ignore_index
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    shifted = flat_logits - np.max(flat_logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(len(flat_labels)), flat_labels]
    return np.mean(-correct + logsumexp)
```

### 解析

1. 自回归训练要 shift。
2. `ignore_index` 用于屏蔽 prompt 或 padding。
3. flatten 后只对有效位置算 CE。


:::

## 工程要点 / 面试追问

见 `notes.md`。