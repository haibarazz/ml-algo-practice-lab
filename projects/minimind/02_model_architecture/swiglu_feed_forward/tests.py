"""Lightweight tests for swiglu_feed_forward.

Run with: python tests.py
"""

import numpy as np

def swiglu_ffn(x, w_gate, w_up, w_down):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ np.asarray(w_gate, dtype=np.float64)
    up = x @ np.asarray(w_up, dtype=np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ np.asarray(w_down, dtype=np.float64)


import numpy as np


def test_swiglu_ffn():
    x = np.array([[1.0, 2.0]])
    w_gate = np.eye(2)
    w_up = np.eye(2)
    w_down = np.eye(2)
    out = swiglu_ffn(x, w_gate, w_up, w_down)
    expected = x * (x / (1 + np.exp(-x)))
    assert np.allclose(out, expected)


test_swiglu_ffn()
print("All tests passed.")
