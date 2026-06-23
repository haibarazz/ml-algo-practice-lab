# BatchNorm Backward

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/02_dl_basics/batch_norm_backward/batch_norm_backward.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

BatchNorm backward 可以用紧凑公式：

$$dx=\frac{1}{N}\gamma(\sigma^2+\epsilon)^{-1/2}(N dout - \sum dout - \hat x\sum(dout\hat x))$$

其中求和沿 batch 维。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def batch_norm_backward(dout, cache):
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

def batch_norm_backward(dout, cache):
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
def test_batch_norm_backward():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    gamma = np.array([1.0, 1.5])
    beta = np.zeros(2)
    mean = X.mean(axis=0); var = X.var(axis=0); eps = 1e-5
    xhat = (X - mean) / np.sqrt(var + eps)
    cache = {"xhat": xhat, "gamma": gamma, "var": var, "eps": eps}
    dout = np.ones_like(X)
    dx, dgamma, dbeta = batch_norm_backward(dout, cache)
    assert dx.shape == X.shape
    assert np.allclose(dgamma, np.sum(xhat, axis=0))
    assert np.allclose(dbeta, np.array([3.0, 3.0]))
    assert np.allclose(dx.sum(axis=0), np.zeros(2), atol=1e-8)

test_batch_norm_backward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def batch_norm_backward(dout, cache):
    dout = np.asarray(dout, dtype=np.float64)
    xhat = cache["xhat"]
    gamma = cache["gamma"]
    var = cache["var"]
    eps = cache["eps"]
    N = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * xhat, axis=0)
    dxhat = dout * gamma
    dx = (1.0 / N) / np.sqrt(var + eps) * (N * dxhat - dxhat.sum(axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
    return dx, dgamma, dbeta
```

### 解析

1. `dbeta` 是 dout 沿 batch 求和。
2. `dgamma` 是 `dout * xhat` 沿 batch 求和。
3. `dx` 使用紧凑公式，shape 和 X 一致。
4. 这个测试验证 shape 和关键守恒性质。


:::

## 工程要点 / 面试追问

### 核心公式

- 紧凑写法：$dx=\frac{\gamma}{m\sqrt{\sigma^2+\epsilon}}\left(m\,dy-\sum dy-\hat x\sum(dy\hat x)\right)$。
- $d\gamma=\sum_i dy_i\hat x_i$，$d\beta=\sum_i dy_i$。

### 易错点

- 求和 axis 写错，batch 维和 feature 维混淆。
- 忘记乘 gamma，导致传回上一层的梯度尺度错误。
- backward 必须使用 forward 缓存的同一个 `var + eps`。
- 手推链路较长，推荐先写数值梯度校验。

### 面试追问

- 手推 BN backward 的关键链路是什么？
- 为什么 `d_beta` 是上游梯度求和，`d_gamma` 要乘 normalized input？
- BN backward 中哪些量必须从 forward 缓存？
- BN 对 batch size 敏感会怎样影响训练稳定性？