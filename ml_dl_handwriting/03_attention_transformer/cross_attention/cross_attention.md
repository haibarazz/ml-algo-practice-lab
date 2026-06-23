# Cross Attention

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Cross-Attention 中 query 来自解码端或当前状态，key/value 来自另一段上下文。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def cross_attention(query, context, Wq, Wk, Wv):
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

def cross_attention(query, context, Wq, Wk, Wv):
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
def test_cross_attention():
    query = np.array([[1.0, 0.0]])
    context = np.eye(2)
    W = np.eye(2)
    out, weights = cross_attention(query, context, W, W, W)
    assert out.shape == (1, 2)
    assert weights.shape == (1, 2)
    assert np.allclose(weights.sum(axis=1), [1.0])

test_cross_attention()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def cross_attention(query, context, Wq, Wk, Wv):
    Q = query @ Wq
    K = context @ Wk
    V = context @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights
```

### 解析

1. Query 和 context 可以长度不同。
2. K/V 必须来自同一个 context。
3. 输出长度等于 query 长度。

## 工程要点 / 面试追问

### 核心公式

- cross-attention 中 $Q$ 来自当前解码端/查询序列，$K,V$ 来自上下文序列。
- $O=softmax(QK_{ctx}^\top/\sqrt{d_k})V_{ctx}$，输出长度等于 query 长度。

### 易错点

- 误以为输出长度等于 context 长度；实际由 Q 的长度决定。
- Q/K/V 投影来源写混，导致 encoder-decoder 语义错误。
- mask 要区分 query mask、key padding mask 和 causal mask。
- context 长度很长时，cross-attention 也会带来较高显存成本。

### 面试追问

::: details 参考回答：encoder-decoder Transformer 哪些位置使用 cross-attention？

encoder-decoder Transformer 中，decoder 的每一层通常先做 masked self-attention，再通过 cross-attention 读取 encoder 输出。机器翻译、摘要等 seq2seq 模型里，这一步负责把目标端生成和源端表示对齐。

:::

::: details 参考回答：cross-attention 和 self-attention 在 Q/K/V 来源上有什么区别？

self-attention 的 Q/K/V 来自同一序列，建模序列内部关系。cross-attention 的 Q 来自当前查询序列，K/V 来自外部上下文，因此本质是“用 query 去检索 context”。

:::

::: details 参考回答：RAG 或多模态模型中 cross-attention 的直觉是什么？

在 RAG 中，cross-attention 可以让生成端按需读取检索文档表示；在多模态中，文本 query 可以读取图像或音频 token。直觉上它是一个可学习的软检索和信息融合模块。

:::

::: details 参考回答：cross-attention 的输出长度由什么决定？

输出长度由 Q 的长度决定，因为每个 query 位置都会得到一个加权后的 value 表示。K/V 只决定可被读取的上下文范围和注意力分布的 key 维度。

:::
