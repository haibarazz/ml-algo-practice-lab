# Binary Logistic Regression Solution

## Reference Implementation

```python
import numpy as np

def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def logistic_regression_binary(X, y, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        eps = 1e-12
        losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        diff = p - y
        w -= lr * (X.T @ diff / X.shape[0])
        b -= lr * np.mean(diff)
    return w, b, losses
```

## Explanation

1. sigmoid 输入做 clip，避免指数溢出。
2. BCE 里加 eps，避免 log(0)。
3. BCE + sigmoid 的梯度化简为 `p - y`。
4. 测试使用线性可分数据验证方向正确。
