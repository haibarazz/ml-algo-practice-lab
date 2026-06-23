# LayerNorm Forward

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/02_dl_basics/layer_norm_forward/layer_norm_forward.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

LayerNorm 对每个样本的特征维做归一化：

$$\hat x=(x-\mu_{sample})/\sqrt{\sigma^2_{sample}+\epsilon}$$

和 BatchNorm 不同，它不依赖 batch 统计。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
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

def layer_norm_forward(X, gamma, beta, eps=1e-5):
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
def test_layer_norm_forward():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = layer_norm_forward(X, np.ones(3), np.zeros(3))
    assert np.allclose(out.mean(axis=1), np.zeros(2), atol=1e-6)
    assert np.allclose(out.var(axis=1), np.ones(2), atol=1e-5)

test_layer_norm_forward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    xhat = (X - mean) / np.sqrt(var + eps)
    return gamma * xhat + beta
```

### 解析

1. LayerNorm 沿最后一维统计。
2. gamma/beta 通常也是最后一维大小。
3. 它适合变长序列和小 batch 场景。


:::

## 工程要点 / 面试追问

### 核心公式

- $\mu=\frac{1}{d}\sum_j x_j$，$\sigma^2=\frac{1}{d}\sum_j(x_j-\mu)^2$。
- $y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$，统计量通常在每个样本的特征维上计算。

### 易错点

- 和 BatchNorm 混淆 axis；LN 不依赖 batch 维统计。
- 忘记 `keepdims`，gamma/beta 广播容易错。
- normalized_shape 和最后若干维不匹配。
- Pre-LN/Post-LN 的位置改变会影响深层 Transformer 的梯度流。

### 面试追问

::: details 参考回答：LayerNorm 和 BatchNorm 的统计维度有什么区别？

BatchNorm 通常对 batch 维统计每个特征的均值方差，依赖同 batch 样本。LayerNorm 对每个样本内部的特征维统计，不依赖 batch 中其他样本。

:::

::: details 参考回答：为什么 NLP/Transformer 中 LayerNorm 比 BatchNorm 常见？

NLP/Transformer 中 batch size、序列长度和 padding 经常变化，batch 统计不稳定。LayerNorm 按 token 或样本自身归一化，训练和推理行为一致，更适合自回归和小 batch 场景。

:::

::: details 参考回答：Pre-LN 和 Post-LN Transformer 的训练稳定性有什么差异？

Post-LN 把 LayerNorm 放在残差之后，深层时梯度路径更容易不稳定。Pre-LN 把 LayerNorm 放在子层之前，让残差分支提供更直接的梯度通道，因此深层 Transformer 更容易训练。

:::

::: details 参考回答：LayerNorm 的 gamma/beta 是否必须？去掉会怎样？

gamma/beta 不是数学上必须的，但它们提供可学习的尺度和平移，避免归一化过度限制表示。去掉后模型仍可运行，但表达灵活性和最终效果可能下降。

:::