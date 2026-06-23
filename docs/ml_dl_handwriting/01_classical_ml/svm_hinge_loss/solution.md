# SVM Hinge Loss Solution

## Reference Implementation

```python
import numpy as np

def svm_hinge_loss(X, y, w, b=0.0, C=1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    margins = 1 - y * (X @ w + b)
    active = margins > 0
    loss = 0.5 * np.dot(w, w) + C * np.mean(np.maximum(0, margins))
    grad_w = w - C * (X[active].T @ y[active]) / X.shape[0]
    grad_b = -C * np.sum(y[active]) / X.shape[0]
    return loss, grad_w, grad_b
```

## Explanation

1. `margin = 1 - y * score`。
2. 只有 `margin > 0` 的样本进入 hinge 梯度。
3. 正则项梯度是 `w`。
4. 标签约定必须是 `{-1, +1}`。
