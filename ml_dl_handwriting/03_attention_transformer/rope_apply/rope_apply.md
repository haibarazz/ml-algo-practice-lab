# RoPE Apply

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

RoPE 把相邻两维组成二维平面，根据位置角度旋转：

$$[x_1,x_2] \to [x_1\cos\theta-x_2\sin\theta, x_1\sin\theta+x_2\cos\theta]$$

它把相对位置信息注入到 Q/K 的内积中。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def rope_apply(x, base=10000):
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

def rope_apply(x, base=10000):
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
def test_rope_apply():
    x = np.ones((3, 4))
    out = rope_apply(x)
    assert out.shape == x.shape
    assert np.allclose(out[0], x[0])
    pair_norm_in = np.sum(x[:, 0:2] ** 2, axis=1)
    pair_norm_out = np.sum(out[:, 0:2] ** 2, axis=1)
    assert np.allclose(pair_norm_in, pair_norm_out)

test_rope_apply()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def rope_apply(x, base=10000):
    x = np.asarray(x, dtype=np.float64)
    seq_len, dim = x.shape
    assert dim % 2 == 0
    pos = np.arange(seq_len)[:, None]
    freq = 1.0 / (base ** (np.arange(0, dim, 2)[None, :] / dim))
    theta = pos * freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out
```

### 解析

1. RoPE 要求旋转维度为偶数。
2. pos=0 旋转角为 0，输出不变。
3. 旋转保持每个二维 pair 的范数。

## 工程要点 / 面试追问

### 核心公式

- 二维配对旋转：$[x_{2i},x_{2i+1}] \mapsto [x_{2i}\cos\theta-x_{2i+1}\sin\theta,\;x_{2i}\sin\theta+x_{2i+1}\cos\theta]$。
- RoPE 通常作用于 Q/K，使 attention score 依赖相对位置。

### 易错点

- 奇偶维配对错，导致旋转不成对。
- sin/cos 广播维度错，batch/head/seq 维对不上。
- 对 V 也应用 RoPE；通常 RoPE 用于 Q/K。
- position index 从 0 还是从 offset 开始，在 KV cache 推理中必须一致。

### 面试追问

- RoPE 为什么能表达相对位置信息？
- RoPE 和绝对位置编码有什么区别？
- RoPE 在 KV cache 推理时 position offset 如何处理？
- RoPE 和 ALiBi 的设计直觉有什么不同？
