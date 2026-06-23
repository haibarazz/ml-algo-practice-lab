"""Lightweight tests for rope_apply.

Run with: python tests.py
"""

import numpy as np

def rope_apply(x, base=10000):
    x = np.asarray(x, dtype=np.float64)
    seq_len, dim = x.shape
    assert dim % 2 == 0
    pos = np.arange(seq_len)[:, None]
    freq = 1.0 / (base ** (np.arange(0, dim, 2)[None, :] / dim))
    theta = pos * freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


def test_rope_apply():
    x = np.ones((3, 4))
    out = rope_apply(x)
    assert out.shape == x.shape
    assert np.allclose(out[0], x[0])
    pair_norm_in = np.sum(x[:, 0:2] ** 2, axis=1)
    pair_norm_out = np.sum(out[:, 0:2] ** 2, axis=1)
    assert np.allclose(pair_norm_in, pair_norm_out)

test_rope_apply()
print("All tests passed.")
