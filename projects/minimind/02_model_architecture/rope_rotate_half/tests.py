"""Lightweight tests for rope_rotate_half.

Run with: python tests.py
"""

import numpy as np

def apply_rope_rotate_half(x, cos, sin):
    x = np.asarray(x, dtype=np.float64)
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.asarray(cos) + rotated * np.asarray(sin)


import numpy as np


def test_apply_rope_rotate_half():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cos = np.ones(4)
    sin = np.zeros(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), x)
    cos = np.zeros(4)
    sin = np.ones(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), np.array([-3.0, -4.0, 1.0, 2.0]))


test_apply_rope_rotate_half()
print("All tests passed.")
