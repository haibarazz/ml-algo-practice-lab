# BatchNorm Forward

> Status: complete

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

## 工程要点 / 面试追问

### 核心公式

- $\mu_B=\frac{1}{m}\sum_i x_i$，$\sigma_B^2=\frac{1}{m}\sum_i(x_i-\mu_B)^2$。
- $\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$，$y_i=\gamma\hat x_i+\beta$。

### 易错点

- 把 LayerNorm 的 axis 用到 BatchNorm；BN 通常按 batch 维统计每个特征。
- 忘记 eps，方差很小时会除零或放大噪声。
- 训练态用 batch stats，推理态用 running stats，本模块只做训练态。
- batch size 很小时统计不稳定，效果可能变差。

### 面试追问

::: details 参考回答：BatchNorm 训练和推理有什么不同？

训练时 BN 使用当前 mini-batch 的均值和方差，并更新 running mean/var。推理时不能依赖单个 batch 的统计量，而是使用训练期间累积的 running stats 保证输出稳定。

:::

::: details 参考回答：BN 为什么对 batch size 敏感？

BN 的统计量来自 batch 维，batch size 太小时均值和方差估计噪声很大。这样会让归一化本身引入随机扰动，导致训练不稳定或推理统计量不可靠。

:::

::: details 参考回答：BN 的 gamma 和 beta 有什么作用？

gamma 和 beta 让模型在归一化后仍能学习合适的尺度和平移。否则归一化会强行限制表示分布，可能削弱某些层需要保留的幅度信息。

:::

::: details 参考回答：为什么 Transformer/LLM 更常用 LayerNorm 或 RMSNorm？

Transformer/LLM 的序列长度、batch 组成和自回归推理形态让 batch 统计不稳定，也不方便跨 token 使用。LayerNorm/RMSNorm 按样本内部维度归一化，不依赖 batch，更适合变长序列和小 batch 训练。

:::
