"""Lightweight tests for mse_loss.

Run with: python tests.py
"""

import numpy as np

def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    return np.mean(diff * diff)


def test_mse_loss():
    assert np.allclose(mse_loss([1, 2, 3], [1, 2, 5]), 4 / 3)
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = y + 1.0
    assert np.allclose(mse_loss(y, pred), 1.0)

test_mse_loss()
print("All tests passed.")
