# Entropy And KL Solution

## Reference Implementation

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    if q is None:
        return entropy
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sum(q)
    q_safe = np.clip(q, eps, 1.0)
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))
    return entropy, kl
```

## Explanation

1. 输入先归一化，避免调用方给的是未归一化权重。
2. `eps` 裁剪用于避免 `log(0)`。
3. KL 不对称，`KL(p||q)` 和 `KL(q||p)` 通常不同。
