"""Lightweight tests for logistic_regression_multiclass.

Run with: python tests.py
"""

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


def test_logistic_regression_multiclass():
    X = np.array([[2, 0], [0, 2], [-2, -2]], dtype=float)
    y = np.array([0, 1, 2])
    W, b, losses = logistic_regression_multiclass(X, y, 3, lr=0.3, steps=200)
    probs = _softmax(X @ W + b)
    assert losses[-1] < losses[0]
    assert np.argmax(probs, axis=1).tolist() == [0, 1, 2]

test_logistic_regression_multiclass()
print("All tests passed.")
