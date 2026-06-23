# LayerNorm Forward Solution

## Reference Implementation

```python
import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    xhat = (X - mean) / np.sqrt(var + eps)
    return gamma * xhat + beta
```

## Explanation

1. LayerNorm 沿最后一维统计。
2. gamma/beta 通常也是最后一维大小。
3. 它适合变长序列和小 batch 场景。
