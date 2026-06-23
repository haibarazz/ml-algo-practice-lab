"""Lightweight tests for logistic_regression_binary.

Run with: python tests.py
"""

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


def test_logistic_regression_binary():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    w, b, losses = logistic_regression_binary(X, y, lr=0.5, steps=200)
    assert losses[-1] < losses[0]
    probs = _sigmoid(X @ w + b)
    assert (probs[:2] < 0.5).all()
    assert (probs[2:] > 0.5).all()

test_logistic_regression_binary()
print("All tests passed.")
