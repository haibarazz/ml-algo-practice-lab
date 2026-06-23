"""Lightweight tests for mha_with_mask.

Run with: python tests.py
"""

import numpy as np

def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _split_heads(x, num_heads):
    seq, dim = x.shape
    return x.reshape(seq, num_heads, dim // num_heads).transpose(1, 0, 2)

def mha_with_mask(X, Wq, Wk, Wv, Wo, num_heads, mask):
    Q = _split_heads(X @ Wq, num_heads)
    K = _split_heads(X @ Wk, num_heads)
    V = _split_heads(X @ Wv, num_heads)
    heads = []
    weights = []
    for h in range(num_heads):
        scores = Q[h] @ K[h].T / np.sqrt(Q.shape[-1])
        scores = np.where(mask, scores, -1e9)
        w = _softmax(scores, axis=-1)
        heads.append(w @ V[h])
        weights.append(w)
    return np.concatenate(heads, axis=-1) @ Wo, np.asarray(weights)


def test_mha_with_mask():
    X = np.eye(4)
    W = np.eye(4)
    mask = np.tril(np.ones((4, 4), dtype=bool))
    out, weights = mha_with_mask(X, W, W, W, W, 2, mask)
    assert out.shape == (4, 4)
    assert np.allclose(weights[:, 0, 1:], 0.0)
    assert np.allclose(weights.sum(axis=-1), np.ones((2, 4)))

test_mha_with_mask()
print("All tests passed.")
