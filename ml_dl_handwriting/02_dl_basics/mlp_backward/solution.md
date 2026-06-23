# MLP Backward Solution

## Reference Implementation

```python
import numpy as np


def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Forward and backward for a two-layer MLP with ReLU and MSE.

    Returns:
        loss: scalar float
        grads: dict with dW1, db1, dW2, db2
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    pred = a1 @ W2 + b2

    diff = pred - y
    loss = np.mean(diff * diff)

    dpred = 2.0 * diff / diff.size
    dW2 = a1.T @ dpred
    db2 = dpred.sum(axis=0)
    da1 = dpred @ W2.T

    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return loss, grads
```

## Explanation

1. MSE 的平均范围是所有输出元素，因此 `dpred = 2 * diff / diff.size`。
2. 线性层 `Y = A @ W + b` 的梯度是 `dW = A.T @ dY`，`db = sum(dY)`。
3. ReLU backward 用 mask：`z1 > 0` 的位置梯度通过，否则为 0。
4. 所有梯度 shape 必须和参数 shape 完全一致，这是反向传播最基本的检查。
