"""Lightweight tests for cross_attention.

Run with: python tests.py
"""

import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def cross_attention(query, context, Wq, Wk, Wv):
    Q = query @ Wq
    K = context @ Wk
    V = context @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights


def test_cross_attention():
    query = np.array([[1.0, 0.0]])
    context = np.eye(2)
    W = np.eye(2)
    out, weights = cross_attention(query, context, W, W, W)
    assert out.shape == (1, 2)
    assert weights.shape == (1, 2)
    assert np.allclose(weights.sum(axis=1), [1.0])

test_cross_attention()
print("All tests passed.")
