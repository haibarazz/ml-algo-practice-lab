# Self Attention Solution

## Reference Implementation

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

## Explanation

1. 同一个 X 分别投影成 Q/K/V。
2. 后续就是 scaled dot-product attention。
3. 返回权重便于检查每个 token 关注了谁。
