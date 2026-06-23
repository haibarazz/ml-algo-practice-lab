# Scaled Dot Product Attention Solution

## Reference Implementation

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

## Explanation

1. `Q @ K.T` 得到相似度。
2. 除以 `sqrt(d)` 控制 logits 尺度。
3. mask 必须在 softmax 前作用。
4. 权重乘 V 得到上下文向量。
