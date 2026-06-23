# Cross Entropy Solution

## Reference Implementation

```python
import numpy as np

def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(logits.shape[0]), targets]
    return np.mean(-correct + logsumexp)
```

## Explanation

1. 减最大值避免 `exp(1000)` 溢出。
2. `correct` 取出目标类别对应的 shifted logit。
3. 每个样本损失是 `-correct + logsumexp`。
4. 最后对 batch 求平均。
