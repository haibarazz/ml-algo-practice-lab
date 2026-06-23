"""Lightweight tests for linear_regression_gd.

Run with: python tests.py
"""

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


def test_linear_regression_gd():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X[:, 0] + 1
    w, b, losses = linear_regression_gd(X, y, lr=0.05, steps=300)
    assert losses[-1] < losses[0]
    assert np.allclose(w[0], 2.0, atol=0.15)
    assert np.allclose(b, 1.0, atol=0.25)

test_linear_regression_gd()
print("All tests passed.")
