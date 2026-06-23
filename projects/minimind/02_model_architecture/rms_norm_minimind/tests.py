"""Lightweight tests for rms_norm_minimind.

Run with: python tests.py
"""

import numpy as np

def rms_norm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight


import numpy as np


def test_rms_norm():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    weight = np.array([1.0, 2.0])
    out = rms_norm(x, weight, eps=0.0)
    assert np.allclose(out[0], np.array([0.84852814, 2.2627417]), atol=1e-6)
    assert np.allclose(out[1], np.array([0.0, 2.82842712]), atol=1e-6)


test_rms_norm()
print("All tests passed.")
