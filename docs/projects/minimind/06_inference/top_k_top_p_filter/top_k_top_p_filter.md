# Top-k Top-p Filter

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/06_inference/top_k_top_p_filter/top_k_top_p_filter.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `model/model_minimind.py:257-288`

## 手写实现约束

允许 NumPy；返回过滤后的 logits。

## 原理最小说明

生成时先按 top-k 去掉排名靠后的 token，再按 top-p 去掉累计概率超过阈值后的 token，最后从剩余分布采样或 argmax。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """TODO guided implementation."""
    # TODO 1: 复制 logits 避免原地污染
    # TODO 2: top_k 只保留最大 k 个
    # TODO 3: top_p 按降序 softmax 累计过滤
    # TODO 4: 至少保留最高概率 token
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
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
import numpy as np


def test_top_k_top_p_filter():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    out = top_k_top_p_filter(logits, top_k=2, top_p=1.0)
    assert np.isneginf(out[2]) and np.isneginf(out[3])
    out = top_k_top_p_filter(logits, top_k=0, top_p=0.7)
    assert not np.isneginf(out[0])
    assert np.isneginf(out[-1])


test_top_k_top_p_filter()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64).copy()
    if top_k > 0:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        finite = np.isfinite(sorted_logits)
        shifted = sorted_logits[finite] - np.max(sorted_logits[finite])
        probs = np.exp(shifted) / np.exp(shifted).sum()
        remove = np.zeros_like(sorted_logits, dtype=bool)
        finite_indices = np.where(finite)[0]
        cumulative = np.cumsum(probs)
        remove[finite_indices] = cumulative > top_p
        if len(remove) > 0:
            remove[0] = False
        logits[order[remove]] = -np.inf
    return logits
```

### 解析

1. MiniMind 的实现会把超过 top-p 的 mask 右移，保证第一个 token 保留。
2. top_k 和 top_p 同时开时，top_k 先执行。


:::

## 工程要点 / 面试追问

见 `notes.md`。