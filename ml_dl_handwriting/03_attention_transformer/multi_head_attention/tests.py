"""Lightweight tests for multi_head_attention.

Run with: python tests.py
"""

import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _split_heads(x, num_heads):
    seq, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    Q = _split_heads(X @ Wq, num_heads)
    K = _split_heads(X @ Wk, num_heads)
    V = _split_heads(X @ Wv, num_heads)
    heads = []
    weights = []
    for h in range(num_heads):
        scores = Q[h] @ K[h].T / np.sqrt(Q.shape[-1])
        w = _softmax(scores, axis=-1)
        heads.append(w @ V[h])
        weights.append(w)
    concat = np.concatenate(heads, axis=-1)
    return concat @ Wo, np.asarray(weights)


def test_multi_head_attention():
    X = np.eye(4)
    W = np.eye(4)
    out, weights = multi_head_attention(X, W, W, W, W, num_heads=2)
    assert out.shape == (4, 4)
    assert weights.shape == (2, 4, 4)
    assert np.allclose(weights.sum(axis=-1), np.ones((2, 4)))

test_multi_head_attention()
print("All tests passed.")
