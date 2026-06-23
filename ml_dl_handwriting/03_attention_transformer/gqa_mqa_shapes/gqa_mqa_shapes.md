# GQA MQA Shapes

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

::: details 参考回答：MHA、MQA、GQA 的 KV cache 大小差异是什么？

MHA 每个 Q head 都有自己的 K/V head，KV cache 最大；MQA 所有 Q head 共享一组 K/V，cache 最小；GQA 让一组 Q heads 共享一组 K/V，是两者折中。cache 大小近似随 `n_kv_heads` 线性变化。

:::

::: details 参考回答：为什么大模型推理特别关注 KV cache？

自回归推理每生成一个 token，都要读取所有历史层的 K/V。长上下文和大 batch 下，KV cache 会占用大量显存，并且读取带宽会成为主要瓶颈。

:::

::: details 参考回答：MQA 可能带来什么质量损失？GQA 如何折中？

MQA 共享 K/V 会减少不同 head 的信息多样性，可能带来质量下降。GQA 保留多组 K/V，让多个 Q head 分组共享，在显存、带宽和质量之间折中。

:::

::: details 参考回答：repeat_kv 和直接学习更多 KV head 的区别是什么？

`repeat_kv` 是把少量 K/V head 在运行时复制或广播给多个 Q head，本质参数仍少。直接学习更多 K/V head 会增加参数和 cache，但每个 Q head 或组能拥有更独立的 key/value 表示。

:::
