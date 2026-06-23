# GQA MQA Shapes

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/03_attention_transformer/gqa_mqa_shapes/gqa_mqa_shapes.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

MQA/GQA 的关键是 query heads 多于 key/value heads。推理时 K/V cache 更小，但需要把 K/V heads repeat 到 query head 数量参与 attention。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def gqa_mqa_shapes(q, k, v):
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

def gqa_mqa_shapes(q, k, v):
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
def test_gqa_mqa_shapes():
    q = np.zeros((2, 4, 3, 8))
    k = np.ones((2, 2, 3, 8))
    v = np.ones((2, 2, 3, 8)) * 2
    kr, vr = gqa_mqa_shapes(q, k, v)
    assert kr.shape == q.shape
    assert vr.shape == q.shape
    assert np.allclose(kr[:, 0], kr[:, 1])
    assert np.allclose(vr[:, 2], vr[:, 3])

test_gqa_mqa_shapes()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def _repeat_kv(x, target_heads):
    b, kv_heads, seq, dim = x.shape
    assert target_heads % kv_heads == 0
    repeat = target_heads // kv_heads
    return np.repeat(x, repeat, axis=1)

def gqa_mqa_shapes(q, k, v):
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    target_heads = q.shape[1]
    return _repeat_kv(k, target_heads), _repeat_kv(v, target_heads)
```

### 解析

1. 输入约定 `[batch, heads, seq, head_dim]`。
2. query heads 是目标 head 数。
3. K/V heads 通过 repeat 复用。
4. target_heads 必须能整除 kv_heads。


:::

## 工程要点 / 面试追问

### 核心公式

- MHA：$n_{qheads}=n_{kvheads}$；MQA：$n_{kvheads}=1$；GQA：$1<n_{kvheads}<n_{qheads}$。
- KV cache 规模近似正比于 $layers \times seq \times n_{kvheads} \times head\_dim$。

### 易错点

- 在 seq 维 repeat，而不是在 head 维 repeat K/V。
- 忘记 `target_heads` 必须是 `kv_heads` 的整数倍。
- repeat 后没有保持 head 顺序，导致 Q head 对错 KV group。
- 只看参数量而忽略推理时 KV cache 带宽和显存。

### 面试追问

- MHA、MQA、GQA 的 KV cache 大小差异是什么？
- 为什么大模型推理特别关注 KV cache？
- MQA 可能带来什么质量损失？GQA 如何折中？
- repeat_kv 和直接学习更多 KV head 的区别是什么？