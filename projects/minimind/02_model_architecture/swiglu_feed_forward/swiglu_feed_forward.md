# SwiGLU FFN：门控前馈网络

拆解 MiniMind `FeedForward` 中 gate/up/down 三个投影如何组成现代 LLM 常见 FFN。

## 学习目标

- 理解普通 FFN 与 gated FFN 的差别。
- 掌握 SwiGLU 的 gate、up、down 三段结构。
- 理解 FFN 为什么通常是 Transformer 参数大头。

## MiniMind 源码定位

- `model/model_minimind.py:136-146`

## 源码机制详解

MiniMind 的 `FeedForward` 同时有 `gate_proj`、`up_proj` 和 `down_proj`。输入 x 分别经过 gate 和 up 两条分支，gate 分支过激活函数后与 up 分支逐元素相乘，最后再经 down 投影回 hidden size。
这类结构不是简单的 `Linear -> Act -> Linear`。乘法 gate 让模型对每个 token 动态选择哪些隐藏通道通过，表达能力比纯 ReLU/GELU FFN 更强。
配置中的 `intermediate_size` 默认按 `ceil(hidden_size * pi / 64) * 64` 取整，说明 MiniMind 在小模型里也保留了 LLM 常见的 FFN expansion 设计，并对硬件友好的 64 倍数做对齐。

## 关键公式与数据流

- $SwiGLU(x)=W_d\left(SiLU(W_gx)\odot W_ux\right)$。
- $SiLU(z)=z\sigma(z)$。
- 参数主要来自 $W_g,W_u,W_d$ 三个矩阵。

## 为什么练这个

- 手写 SwiGLU 能理解现代 LLM FFN 为什么不是普通 MLP。
- 练习重点是两条分支的 shape 必须一致，以及输出要回到 hidden size。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def swiglu_ffn(x, w_gate, w_up, w_down):
    """带提示实现。"""
    # TODO 1: 实现 silu
    # TODO 2: 计算 gate/up 两条分支
    # TODO 3: 逐元素相乘
    # TODO 4: 再乘 down 投影
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def swiglu_ffn(x, w_gate, w_up, w_down):
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


def test_swiglu_ffn():
    x = np.array([[1.0, 2.0]])
    w_gate = np.eye(2)
    w_up = np.eye(2)
    w_down = np.eye(2)
    out = swiglu_ffn(x, w_gate, w_up, w_down)
    expected = x * (x / (1 + np.exp(-x)))
    assert np.allclose(out, expected)


test_swiglu_ffn()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def swiglu_ffn(x, w_gate, w_up, w_down):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ np.asarray(w_gate, dtype=np.float64)
    up = x @ np.asarray(w_up, dtype=np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ np.asarray(w_down, dtype=np.float64)
```

### 解析

1. 如果 hidden_act 换掉，gate 分支的激活也会变。
2. 三个矩阵 shape 要能完成 hidden -> intermediate -> hidden。

## 工程要点 / 面试追问

### 关键公式与数据流

- $SwiGLU(x)=W_d\left(SiLU(W_gx)\odot W_ux\right)$。
- $SiLU(z)=z\sigma(z)$。
- 参数主要来自 $W_g,W_u,W_d$ 三个矩阵。

### 易错点

- 把 gate 分支写成普通 sigmoid GLU，会和 MiniMind 的 hidden_act 不一致。
- 忘记 down projection 会让输出维度无法接回残差。
- gate/up 维度不一致时逐元素乘法会隐式广播或直接报错。

### 面试追问

::: details 参考回答：SwiGLU 为什么常见于现代 LLM？

它用数据依赖的门控控制隐藏通道，比单一激活函数更灵活。经验上在相近计算预算下，gated FFN 往往比 ReLU/GELU FFN 有更好的表达和训练效果。

:::

::: details 参考回答：FFN 在 Transformer 中起什么作用？

attention 负责 token 间信息交换，FFN 负责对每个 token 的表示做非线性变换。它是逐 token 独立计算的，但参数量通常占整个 block 的很大比例。

:::
