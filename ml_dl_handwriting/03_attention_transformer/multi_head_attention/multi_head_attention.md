# Multi Head Attention

> Status: complete

## 题源线索

- Topic: Multi-Head Attention。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

MHA 把模型维度拆成多个 head，每个 head 独立做 attention，再 concat 回来接输出投影。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
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

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
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
def test_multi_head_attention():
    X = np.eye(4)
    W = np.eye(4)
    out, weights = multi_head_attention(X, W, W, W, W, num_heads=2)
    assert out.shape == (4, 4)
    assert weights.shape == (2, 4, 4)
    assert np.allclose(weights.sum(axis=-1), np.ones((2, 4)))

test_multi_head_attention()
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

def _split_heads(x, num_heads):
    seq, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    Q = _split_heads(X @ Wq, num_heads)
    K = _split_heads(X @ Wk, num_heads)
    V = _split_heads(X @ Wv, num_heads)
    heads = []
    weights = []
    for h in range(num_heads):
        scores = Q[h] @ K[h].T / np.sqrt(Q.shape[-1])
        w = _softmax(scores, axis=-1)
        heads.append(w @ V[h])
        weights.append(w)
    concat = np.concatenate(heads, axis=-1)
    return concat @ Wo, np.asarray(weights)
```

### 解析

1. `dim` 必须能被 `num_heads` 整除。
2. split 后每个 head shape 是 `[seq, head_dim]`。
3. concat 后恢复到模型维度。
4. 最后接输出投影 `Wo`。

## 工程要点 / 面试追问

见 `notes.md`。
