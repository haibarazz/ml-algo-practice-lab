"""Lightweight tests for svm_hinge_loss.

Run with: python tests.py
"""

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


def test_svm_hinge_loss():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    loss, gw, gb = svm_hinge_loss(X, y, np.array([0.0, 0.0]), 0.0, C=1.0)
    assert np.allclose(loss, 1.0)
    assert np.allclose(gw, np.array([-0.5, 0.5]))
    assert np.allclose(gb, 0.0)

test_svm_hinge_loss()
print("All tests passed.")
