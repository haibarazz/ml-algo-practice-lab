"""Lightweight tests for batch_norm_forward.

Run with: python tests.py
"""

import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    xhat = (X - mean) / np.sqrt(var + eps)
    out = gamma * xhat + beta
    cache = {"X": X, "mean": mean, "var": var, "xhat": xhat, "gamma": gamma, "eps": eps}
    return out, cache


def test_batch_norm_forward():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out, cache = batch_norm_forward(X, np.ones(2), np.zeros(2))
    assert np.allclose(out.mean(axis=0), np.zeros(2), atol=1e-6)
    assert np.allclose(out.var(axis=0), np.ones(2), atol=1e-5)
    assert "xhat" in cache

test_batch_norm_forward()
print("All tests passed.")
