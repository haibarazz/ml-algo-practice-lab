# Causal LM Shift Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/03_pretraining/causal_lm_shift_loss/causal_lm_shift_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `model/model_minimind.py:245-253`
- `trainer/train_pretrain.py:24-80`

## 手写实现约束

允许 NumPy；忽略标签 `-100`。

## 原理最小说明

Causal LM 用当前位置 logits 预测下一个 token，因此 logits 去掉最后一位，labels 去掉第一位，然后 flatten 做交叉熵。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    """TODO guided implementation."""
    # TODO 1: shift logits 和 labels
    # TODO 2: flatten batch/seq
    # TODO 3: 过滤 ignore_index
    # TODO 4: 稳定 softmax 交叉熵
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def causal_lm_shift_loss(logits, labels, ignore_index=-100):
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


def test_causal_lm_shift_loss():
    logits = np.array([[[0, 5, 0], [0, 0, 5], [5, 0, 0]]], dtype=float)
    labels = np.array([[0, 1, 2]])
    assert causal_lm_shift_loss(logits, labels) < 0.02
    masked = np.array([[0, -100, 2]])
    assert np.isfinite(causal_lm_shift_loss(logits, masked))


test_causal_lm_shift_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    x = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    y = labels[:, 1:].reshape(-1)
    mask = y != ignore_index
    x = x[mask]
    y = y[mask]
    shifted = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    correct = shifted[np.arange(len(y)), y]
    return float(np.mean(-correct + logsumexp))
```

### 解析

1. logits 和 labels 的错位是最常见 bug。
2. ignore_index 是为了屏蔽 pad 或非 assistant token。


:::

## 工程要点 / 面试追问

### Source Mapping

- `model/model_minimind.py:245-253`
- `trainer/train_pretrain.py:24-80`

### 常见坑

- logits 和 labels 的错位是最常见 bug。
- ignore_index 是为了屏蔽 pad 或非 assistant token。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？