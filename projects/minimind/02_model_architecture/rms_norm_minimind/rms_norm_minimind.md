# RMSNorm：只按均方根缩放的归一化

拆解 MiniMind 中 `RMSNorm` 的公式、数值稳定项和它在 Transformer block 中的位置。

## 学习目标

- 掌握 RMSNorm 与 LayerNorm 的差异。
- 理解 eps、float32 计算和 weight 缩放的工程意义。
- 知道 RMSNorm 在 Pre-LN Transformer block 中怎样稳定残差流。

## MiniMind 源码定位

- `model/model_minimind.py:50-60`
- `model/model_minimind.py:178-204`

## 源码机制详解

MiniMind 的 `RMSNorm.norm` 不减均值，只计算最后一维 `x.pow(2).mean(-1)`，再乘 `rsqrt(mean + eps)`。这保留了输入的均值偏移，只消除整体尺度变化。
`forward` 先把输入转成 float 做归一化，再 cast 回原 dtype。这是混合精度训练中的常见写法：归一化统计对精度敏感，用 float32 更稳，而输出仍保持模型计算 dtype。
在 `MiniMindBlock` 中，attention 前和 FFN 前各有一次 RMSNorm，残差连接在子层输出后相加。这是 Pre-Norm 风格，能让深层网络的梯度路径更直接。

## 关键公式与数据流

- $RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}$。
- $RMSNorm(x)=w\odot \frac{x}{RMS(x)}$。
- 与 LayerNorm 相比，RMSNorm 去掉 $x-\mu$，只保留尺度归一化。

## 为什么练这个

- 手写 RMSNorm 是理解现代 LLM 归一化层的入口。
- 它比 LayerNorm 更短，但能暴露 dtype、eps、broadcast 等工程细节。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """带提示实现。"""
    # TODO 1: 沿最后一维求 mean(x^2)
    # TODO 2: 加 eps 后 rsqrt
    # TODO 3: 乘回 x
    # TODO 4: 乘 weight 广播
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """无提示实现。"""
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


def test_rms_norm():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    weight = np.array([1.0, 2.0])
    out = rms_norm(x, weight, eps=0.0)
    assert np.allclose(out[0], np.array([0.84852814, 2.2627417]), atol=1e-6)
    assert np.allclose(out[1], np.array([0.0, 2.82842712]), atol=1e-6)


test_rms_norm()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def rms_norm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight
```

### 解析

1. RMSNorm 的 eps 是数值稳定项。
2. weight shape 通常等于 hidden_size。

## 工程要点 / 面试追问

### 关键公式与数据流

- $RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}$。
- $RMSNorm(x)=w\odot \frac{x}{RMS(x)}$。
- 与 LayerNorm 相比，RMSNorm 去掉 $x-\mu$，只保留尺度归一化。

### 易错点

- 把 RMSNorm 写成 LayerNorm，额外减均值会改变模型行为。
- eps 放错位置会影响数值尺度。
- weight shape 应匹配 hidden size 或 head dim，广播轴不能错。

### 面试追问

::: details 参考回答：为什么很多 LLM 使用 RMSNorm 而不是 BatchNorm？

BatchNorm 依赖 batch 统计，训练和自回归推理时分布不一致；RMSNorm 对每个 token 的 hidden 维归一化，不依赖 batch，适合变长序列和小 batch。

:::

::: details 参考回答：RMSNorm 相比 LayerNorm 少了什么不变性？

它保留尺度不变性，但不减均值，所以不具备平移不变性。输入整体加一个常数会改变 RMSNorm 输出，而 LayerNorm 会先去中心化。

:::
