# Adam Optimizer Solution

## Reference Implementation

```python
import numpy as np

def adam_optimizer(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_param, m, v
```

## Explanation

1. `m` 是梯度指数滑动平均。
2. `v` 是梯度平方指数滑动平均。
3. 第几步 `t` 用于 bias correction。
4. 更新方向由 `m_hat / sqrt(v_hat)` 决定。
