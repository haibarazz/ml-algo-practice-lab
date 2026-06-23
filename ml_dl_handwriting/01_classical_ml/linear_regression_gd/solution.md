# Linear Regression GD Solution

## Reference Implementation

```python
import numpy as np

def linear_regression_gd(X, y, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    losses = []
    for _ in range(steps):
        pred = X @ w + b
        diff = pred - y
        losses.append(np.mean(diff * diff))
        grad_w = 2.0 * X.T @ diff / X.shape[0]
        grad_b = 2.0 * np.mean(diff)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, losses
```

## Explanation

1. 参数从零初始化即可。
2. 每步先 forward，再算梯度，再更新。
3. 学习率过大会震荡，过小会慢。
4. 测试不要求精确闭式解，只验证收敛到附近。
