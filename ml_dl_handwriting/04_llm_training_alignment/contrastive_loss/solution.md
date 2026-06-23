# Contrastive Loss Solution

## Reference Implementation

```python
import numpy as np

def _normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def contrastive_loss(query, positive, negatives, temperature=1.0):
    q = _normalize(np.asarray(query, dtype=np.float64))
    p = _normalize(np.asarray(positive, dtype=np.float64))
    n = _normalize(np.asarray(negatives, dtype=np.float64))
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum('bd,bkd->bk', q, n)
    logits = np.concatenate([pos, neg], axis=1) / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return np.mean(-log_probs[:, 0])
```

## Explanation

1. 先归一化，用 cosine similarity。
2. 正样本 logit 放第 0 列。
3. 对候选集合做 cross entropy。
4. temperature 越小，分布越尖锐。
