"""Lightweight tests for batch_norm_backward.

Run with: python tests.py
"""

import numpy as np

def batch_norm_backward(dout, cache):
    dout = np.asarray(dout, dtype=np.float64)
    xhat = cache["xhat"]
    gamma = cache["gamma"]
    var = cache["var"]
    eps = cache["eps"]
    N = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * xhat, axis=0)
    dxhat = dout * gamma
    dx = (1.0 / N) / np.sqrt(var + eps) * (N * dxhat - dxhat.sum(axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
    return dx, dgamma, dbeta


def test_batch_norm_backward():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    gamma = np.array([1.0, 1.5])
    beta = np.zeros(2)
    mean = X.mean(axis=0); var = X.var(axis=0); eps = 1e-5
    xhat = (X - mean) / np.sqrt(var + eps)
    cache = {"xhat": xhat, "gamma": gamma, "var": var, "eps": eps}
    dout = np.ones_like(X)
    dx, dgamma, dbeta = batch_norm_backward(dout, cache)
    assert dx.shape == X.shape
    assert np.allclose(dgamma, np.sum(xhat, axis=0))
    assert np.allclose(dbeta, np.array([3.0, 3.0]))
    assert np.allclose(dx.sum(axis=0), np.zeros(2), atol=1e-8)

test_batch_norm_backward()
print("All tests passed.")
