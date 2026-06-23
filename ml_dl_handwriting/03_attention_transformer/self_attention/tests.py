"""Lightweight tests for self_attention.

Run with: python tests.py
"""

import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights


def test_self_attention():
    X = np.eye(2)
    W = np.eye(2)
    out, weights = self_attention(X, W, W, W)
    assert out.shape == X.shape
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), np.ones(2))

test_self_attention()
print("All tests passed.")
