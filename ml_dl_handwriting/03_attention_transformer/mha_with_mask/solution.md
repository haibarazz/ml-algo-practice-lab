# MHA With Mask Solution

## Reference Implementation

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

## Explanation

1. mask shape 是 `[seq, seq]`，会应用到每个 head。
2. False 位置替换成 `-1e9`。
3. softmax 后被 mask 的位置概率接近 0。
