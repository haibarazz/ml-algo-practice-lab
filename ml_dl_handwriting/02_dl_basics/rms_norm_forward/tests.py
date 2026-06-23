"""Lightweight tests for rms_norm_forward.

Run with: python tests.py
"""

import numpy as np

def rms_norm_forward(X, gamma, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)
    rms = np.sqrt(np.mean(X * X, axis=-1, keepdims=True) + eps)
    return X / rms * gamma


def test_rms_norm_forward():
    X = np.array([[3.0, 4.0], [1.0, 1.0]])
    out = rms_norm_forward(X, np.ones(2), eps=0.0)
    assert np.allclose(np.sqrt(np.mean(out * out, axis=1)), np.ones(2))

test_rms_norm_forward()
print("All tests passed.")
