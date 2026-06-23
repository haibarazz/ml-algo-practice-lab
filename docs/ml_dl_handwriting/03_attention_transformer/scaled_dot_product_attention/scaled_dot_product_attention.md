# Scaled Dot Product Attention

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/03_attention_transformer/scaled_dot_product_attention/scaled_dot_product_attention.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: Scaled Dot-Product Attention。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

注意力核心：

$$Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V$$

mask 位置应在 softmax 前置为一个很小的值。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
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

def scaled_dot_product_attention(Q, K, V, mask=None):
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
def test_scaled_dot_product_attention():
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (1, 2)
    assert weights[0, 0] > weights[0, 1]
    masked_out, masked_w = scaled_dot_product_attention(Q, K, V, mask=np.array([[True, False]]))
    assert np.allclose(masked_w, np.array([[1.0, 0.0]]))
    assert np.allclose(masked_out, np.array([[10.0, 0.0]]))

test_scaled_dot_product_attention()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights
```

### 解析

1. `Q @ K.T` 得到相似度。
2. 除以 `sqrt(d)` 控制 logits 尺度。
3. mask 必须在 softmax 前作用。
4. 权重乘 V 得到上下文向量。


:::

## 工程要点 / 面试追问

见 `notes.md`。