# MHA With Mask

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

带 mask 的 MHA 在每个 head 的 score 上屏蔽不可见位置。Causal mask 常用于自回归生成。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def mha_with_mask(X, Wq, Wk, Wv, Wo, num_heads, mask):
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

def mha_with_mask(X, Wq, Wk, Wv, Wo, num_heads, mask):
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
def test_mha_with_mask():
    X = np.eye(4)
    W = np.eye(4)
    mask = np.tril(np.ones((4, 4), dtype=bool))
    out, weights = mha_with_mask(X, W, W, W, W, 2, mask)
    assert out.shape == (4, 4)
    assert np.allclose(weights[:, 0, 1:], 0.0)
    assert np.allclose(weights.sum(axis=-1), np.ones((2, 4)))

test_mha_with_mask()
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
    return x.reshape(seq, num_heads, dim // num_heads).transpose(1, 0, 2)

def mha_with_mask(X, Wq, Wk, Wv, Wo, num_heads, mask):
    Q = _split_heads(X @ Wq, num_heads)
    K = _split_heads(X @ Wk, num_heads)
    V = _split_heads(X @ Wv, num_heads)
    heads = []
    weights = []
    for h in range(num_heads):
        scores = Q[h] @ K[h].T / np.sqrt(Q.shape[-1])
        scores = np.where(mask, scores, -1e9)
        w = _softmax(scores, axis=-1)
        heads.append(w @ V[h])
        weights.append(w)
    return np.concatenate(heads, axis=-1) @ Wo, np.asarray(weights)
```

### 解析

1. mask shape 是 `[seq, seq]`，会应用到每个 head。
2. False 位置替换成 `-1e9`。
3. softmax 后被 mask 的位置概率接近 0。

## 工程要点 / 面试追问

### 核心公式

- $scores=QK^\top/\sqrt{d_k}+mask$，其中被屏蔽位置通常为 $-\infty$。
- $weights=softmax(scores)$，$O=weightsV$。

### 易错点

- mask 语义反了：True 是保留还是屏蔽必须在实现中固定。
- 在 softmax 后 mask，概率和不再为 1。
- mask shape 没有正确广播到 batch/head/query/key 维。
- padding mask 和 causal mask 叠加时 dtype、shape 容易错。

### 面试追问

- causal mask 和 padding mask 的区别是什么？
- 训练和自回归推理时 mask 有什么不同？
- 为什么 mask 应该作用在 softmax 前？
- 如果一整行都被 mask，softmax 会出现什么问题？
