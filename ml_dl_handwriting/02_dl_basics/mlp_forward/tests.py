"""Lightweight tests for mlp_forward.

Run with: python tests.py
"""

import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    X = np.asarray(X, dtype=np.float64)
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    out = a1 @ W2 + b2
    return out, {"z1": z1, "a1": a1}


def test_mlp_forward():
    X = np.array([[1.0, 2.0]])
    W1 = np.array([[1.0, -1.0], [0.5, 2.0]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0], [2.0]])
    b2 = np.array([0.1])
    out, cache = mlp_forward(X, W1, b1, W2, b2)
    assert np.allclose(cache["a1"], np.array([[2.0, 3.0]]))
    assert np.allclose(out, np.array([[8.1]]))

test_mlp_forward()
print("All tests passed.")
