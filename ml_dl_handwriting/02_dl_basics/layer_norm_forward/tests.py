"""Lightweight tests for layer_norm_forward.

Run with: python tests.py
"""

import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    xhat = (X - mean) / np.sqrt(var + eps)
    return gamma * xhat + beta


def test_layer_norm_forward():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = layer_norm_forward(X, np.ones(3), np.zeros(3))
    assert np.allclose(out.mean(axis=1), np.zeros(2), atol=1e-6)
    assert np.allclose(out.var(axis=1), np.ones(2), atol=1e-5)

test_layer_norm_forward()
print("All tests passed.")
