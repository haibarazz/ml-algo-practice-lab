# Top-k Top-p Filter Solution

## Reference Implementation

```python
import numpy as np

def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64).copy()
    if top_k > 0:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        finite = np.isfinite(sorted_logits)
        shifted = sorted_logits[finite] - np.max(sorted_logits[finite])
        probs = np.exp(shifted) / np.exp(shifted).sum()
        remove = np.zeros_like(sorted_logits, dtype=bool)
        finite_indices = np.where(finite)[0]
        cumulative = np.cumsum(probs)
        remove[finite_indices] = cumulative > top_p
        if len(remove) > 0:
            remove[0] = False
        logits[order[remove]] = -np.inf
    return logits
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
