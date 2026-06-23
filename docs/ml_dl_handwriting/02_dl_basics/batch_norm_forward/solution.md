# BatchNorm Forward Solution

## Reference Implementation

```python
import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    xhat = (X - mean) / np.sqrt(var + eps)
    out = gamma * xhat + beta
    cache = {"X": X, "mean": mean, "var": var, "xhat": xhat, "gamma": gamma, "eps": eps}
    return out, cache
```

## Explanation

1. 训练态使用当前 batch 统计量。
2. mean/var 沿 batch 维，也就是 axis=0。
3. gamma/beta 是逐特征仿射参数。
