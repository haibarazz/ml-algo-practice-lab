"""Lightweight tests for swiglu.

Run with: python tests.py
"""

import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def swiglu(X, W_gate, b_gate, W_up, b_up, W_down, b_down):
    gate = X @ W_gate + b_gate
    up = X @ W_up + b_up
    silu_gate = gate * _sigmoid(gate)
    return (silu_gate * up) @ W_down + b_down


def test_swiglu():
    X = np.array([[1.0, 2.0]])
    Wg = np.ones((2, 3)); bg = np.zeros(3)
    Wu = np.ones((2, 3)); bu = np.zeros(3)
    Wd = np.ones((3, 1)); bd = np.zeros(1)
    out = swiglu(X, Wg, bg, Wu, bu, Wd, bd)
    assert out.shape == (1, 1)
    assert out[0, 0] > 0

test_swiglu()
print("All tests passed.")
