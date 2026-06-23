# Cross Attention Solution

## Reference Implementation

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

## Explanation

1. Query 和 context 可以长度不同。
2. K/V 必须来自同一个 context。
3. 输出长度等于 query 长度。
