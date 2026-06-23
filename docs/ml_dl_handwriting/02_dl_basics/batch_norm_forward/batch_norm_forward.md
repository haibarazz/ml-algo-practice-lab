# BatchNorm Forward

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/02_dl_basics/batch_norm_forward/batch_norm_forward.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: BatchNorm 训练态 forward。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

BatchNorm 对 batch 维度统计每个特征的均值和方差：

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$$
$$y=\gamma\hat x+\beta$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
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

def batch_norm_forward(X, gamma, beta, eps=1e-5):
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
def test_batch_norm_forward():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out, cache = batch_norm_forward(X, np.ones(2), np.zeros(2))
    assert np.allclose(out.mean(axis=0), np.zeros(2), atol=1e-6)
    assert np.allclose(out.var(axis=0), np.ones(2), atol=1e-5)
    assert "xhat" in cache

test_batch_norm_forward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    xhat = (X - mean) / np.sqrt(var + eps)
    out = gamma * xhat + beta
    cache = {"X": X, "mean": mean, "var": var, "xhat": xhat, "gamma": gamma, "eps": eps}
    return out, cache
```

### 解析

1. 训练态使用当前 batch 统计量。
2. mean/var 沿 batch 维，也就是 axis=0。
3. gamma/beta 是逐特征仿射参数。


:::

## 工程要点 / 面试追问

见 `notes.md`。