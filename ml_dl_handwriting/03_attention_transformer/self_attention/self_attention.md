# Self Attention

> Status: complete

## 题源线索

- Topic: Self-Attention 封装。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Self-Attention 中 Q/K/V 都来自同一个输入 X，只是经过不同线性投影。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def self_attention(X, Wq, Wk, Wv):
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

def self_attention(X, Wq, Wk, Wv):
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
def test_self_attention():
    X = np.eye(2)
    W = np.eye(2)
    out, weights = self_attention(X, W, W, W)
    assert out.shape == X.shape
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), np.ones(2))

test_self_attention()
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

def self_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights
```

### 解析

1. 同一个 X 分别投影成 Q/K/V。
2. 后续就是 scaled dot-product attention。
3. 返回权重便于检查每个 token 关注了谁。

## 工程要点 / 面试追问

见 `notes.md`。
