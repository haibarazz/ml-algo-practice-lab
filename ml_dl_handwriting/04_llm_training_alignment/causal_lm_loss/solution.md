# Causal LM Loss Solution

## Reference Implementation

```python
import numpy as np

def causal_lm_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    mask = flat_labels != ignore_index
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    shifted = flat_logits - np.max(flat_logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(len(flat_labels)), flat_labels]
    return np.mean(-correct + logsumexp)
```

## Explanation

1. 自回归训练要 shift。
2. `ignore_index` 用于屏蔽 prompt 或 padding。
3. flatten 后只对有效位置算 CE。
