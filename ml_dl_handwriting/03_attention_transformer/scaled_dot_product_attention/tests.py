"""Lightweight tests for scaled_dot_product_attention.

Run with: python tests.py
"""

import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores, axis=-1)
    return weights @ V, weights


def test_scaled_dot_product_attention():
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0]])
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (1, 2)
    assert weights[0, 0] > weights[0, 1]
    masked_out, masked_w = scaled_dot_product_attention(Q, K, V, mask=np.array([[True, False]]))
    assert np.allclose(masked_w, np.array([[1.0, 0.0]]))
    assert np.allclose(masked_out, np.array([[10.0, 0.0]]))

test_scaled_dot_product_attention()
print("All tests passed.")
