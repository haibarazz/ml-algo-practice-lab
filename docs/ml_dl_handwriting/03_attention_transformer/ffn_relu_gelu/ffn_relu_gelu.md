# FFN Relu Gelu

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/03_attention_transformer/ffn_relu_gelu/ffn_relu_gelu.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Transformer FFN 是逐 token 的两层 MLP：

$$FFN(x)=W_2 activation(W_1x+b_1)+b_2$$

常见激活包括 ReLU 和 GeLU。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
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

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
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
def test_ffn_relu_gelu():
    X = np.array([[1.0, -1.0]])
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.ones((2, 1))
    b2 = np.array([0.0])
    assert np.allclose(ffn_relu_gelu(X, W1, b1, W2, b2, 'relu'), np.array([[1.0]]))
    out = ffn_relu_gelu(X, W1, b1, W2, b2, 'gelu')
    assert out.shape == (1, 1)

test_ffn_relu_gelu()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
    hidden = X @ W1 + b1
    if activation == 'relu':
        hidden = np.maximum(hidden, 0.0)
    elif activation == 'gelu':
        hidden = _gelu(hidden)
    else:
        raise ValueError("activation must be relu or gelu")
    return hidden @ W2 + b2
```

### 解析

1. FFN 对每个 token 独立作用。
2. ReLU 直接截断负值。
3. GeLU 常用 tanh 近似。
4. 最后一层投回模型维度。


:::

## 工程要点 / 面试追问

### 核心公式

- Transformer FFN 常见形式：$FFN(x)=W_2\phi(W_1x+b_1)+b_2$。
- GELU 常用近似：$GELU(x)\approx0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$。

### 易错点

- 把 FFN 和 attention 混在一起；FFN 是逐 token 的非线性变换。
- 忘记第二个线性层，输出维度无法回到 `d_model`。
- GeLU 公式写错，或把输入当成 sigmoid gate。
- FFN expansion ratio 会显著影响参数量和计算量。

### 面试追问

::: details 参考回答：Transformer 中 FFN 的参数量通常占比如何？

Transformer 中 FFN 往往占很大参数比例，因为它包含从 `d_model` 到更大 hidden size 再回到 `d_model` 的两层线性变换。常见 expansion ratio 为 4 时，FFN 参数量通常超过 attention 投影。

:::

::: details 参考回答：FFN 为什么可以逐 token 并行计算？

FFN 对每个 token 独立应用同一组线性层和激活，不依赖其他 token。token 间信息交换已经由 attention 完成，所以 FFN 可以在序列维上完全并行。

:::

::: details 参考回答：GELU 相比 ReLU 的直觉优势是什么？

GELU 是平滑的概率式门控，输入越大通过越多，输入较小时被柔和抑制。相比 ReLU 的硬截断，它在 Transformer 中常带来更平滑的优化和更好的表达。

:::

::: details 参考回答：FFN hidden size 为什么常设为 `4 * d_model` 或相近比例？

`4 * d_model` 是经验上兼顾容量和计算的宽度设置，给逐 token 非线性变换足够的中间维度。现代 LLM 使用 SwiGLU 等门控 FFN 时，hidden size 常会调整到约 `8/3 * d_model` 以控制参数量。

:::