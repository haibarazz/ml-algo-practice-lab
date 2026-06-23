# BatchNorm Backward Solution

## Reference Implementation

```python
import numpy as np

def batch_norm_backward(dout, cache):
    dout = np.asarray(dout, dtype=np.float64)
    xhat = cache["xhat"]
    gamma = cache["gamma"]
    var = cache["var"]
    eps = cache["eps"]
    N = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * xhat, axis=0)
    dxhat = dout * gamma
    dx = (1.0 / N) / np.sqrt(var + eps) * (N * dxhat - dxhat.sum(axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
    return dx, dgamma, dbeta
```

## Explanation

1. `dbeta` 是 dout 沿 batch 求和。
2. `dgamma` 是 `dout * xhat` 沿 batch 求和。
3. `dx` 使用紧凑公式，shape 和 X 一致。
4. 这个测试验证 shape 和关键守恒性质。
