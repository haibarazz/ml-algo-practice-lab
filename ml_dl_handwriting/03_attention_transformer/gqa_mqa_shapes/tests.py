"""Lightweight tests for gqa_mqa_shapes.

Run with: python tests.py
"""

import numpy as np

def _repeat_kv(x, target_heads):
    b, kv_heads, seq, dim = x.shape
    assert target_heads % kv_heads == 0
    repeat = target_heads // kv_heads
    return np.repeat(x, repeat, axis=1)

def gqa_mqa_shapes(q, k, v):
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    target_heads = q.shape[1]
    return _repeat_kv(k, target_heads), _repeat_kv(v, target_heads)


def test_gqa_mqa_shapes():
    q = np.zeros((2, 4, 3, 8))
    k = np.ones((2, 2, 3, 8))
    v = np.ones((2, 2, 3, 8)) * 2
    kr, vr = gqa_mqa_shapes(q, k, v)
    assert kr.shape == q.shape
    assert vr.shape == q.shape
    assert np.allclose(kr[:, 0], kr[:, 1])
    assert np.allclose(vr[:, 2], vr[:, 3])

test_gqa_mqa_shapes()
print("All tests passed.")
