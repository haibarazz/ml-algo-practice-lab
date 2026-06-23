# Multiclass Logistic Regression Solution

## Reference Implementation

```python
import numpy as np

def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def logistic_regression_multiclass(X, y, num_classes, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    W = np.zeros((X.shape[1], num_classes))
    b = np.zeros(num_classes)
    losses = []
    for _ in range(steps):
        probs = _softmax(X @ W + b)
        losses.append(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
        grad = probs
        grad[np.arange(len(y)), y] -= 1.0
        grad /= X.shape[0]
        W -= lr * (X.T @ grad)
        b -= lr * grad.sum(axis=0)
    return W, b, losses
```

## Explanation

1. logits 形状是 `[N, C]`。
2. 先 softmax，再取目标类别概率计算 CE。
3. 梯度用 `probs - one_hot`。
4. 注意不要永久污染 `probs`，如果后面还要用原概率应 copy。
