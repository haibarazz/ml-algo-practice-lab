# Sinusoidal Position Encoding

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/03_attention_transformer/sinusoidal_position_encoding/sinusoidal_position_encoding.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Transformer 原始绝对位置编码：

$$PE(pos,2i)=\sin(pos/10000^{2i/d})$$
$$PE(pos,2i+1)=\cos(pos/10000^{2i/d})$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
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

def sinusoidal_position_encoding(seq_len, d_model):
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
def test_sinusoidal_position_encoding():
    pe = sinusoidal_position_encoding(3, 4)
    assert pe.shape == (3, 4)
    assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert not np.allclose(pe[1], pe[2])

test_sinusoidal_position_encoding()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / (10000 ** (i / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    return pe
```

### 解析

1. 偶数维用 sin，奇数维用 cos。
2. pos=0 时 sin 为 0，cos 为 1。
3. 不同频率覆盖不同尺度的位置变化。


:::

## 工程要点 / 面试追问

### 核心公式

- $PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}})$。
- $PE_{pos,2i+1}=\cos(pos/10000^{2i/d_{model}})$。

### 易错点

- 指数写成 `i/d` 还是 `2i/d` 混乱，本实现按偶数维索引配对。
- 奇数 `d_model` 时 cos 维度切片要对齐。
- 位置编码要和 token embedding shape 对齐后相加。
- 绝对位置编码外推长度时不一定和训练分布一致。

### 面试追问

- Transformer 为什么需要位置编码？
- 正弦位置编码为什么使用不同频率？
- 绝对位置编码、可学习位置编码、RoPE 有什么区别？
- 正弦位置编码为什么被认为具备一定长度外推能力？