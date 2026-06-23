# Causal LM Shift Loss Solution

## Reference Implementation

```python
import numpy as np

def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    x = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    y = labels[:, 1:].reshape(-1)
    mask = y != ignore_index
    x = x[mask]
    y = y[mask]
    shifted = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    correct = shifted[np.arange(len(y)), y]
    return float(np.mean(-correct + logsumexp))
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
