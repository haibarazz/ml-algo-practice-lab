# Cross Attention

> Status: complete

## 题源线索

- Topic: Cross-Attention。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

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

见 `notes.md`。
