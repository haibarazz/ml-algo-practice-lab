"""Lightweight tests for relu_sigmoid_tanh.

Run with: python tests.py
"""

import numpy as np

def relu_sigmoid_tanh(x):
    x = np.asarray(x, dtype=np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    tanh = np.tanh(x)
    relu = np.maximum(x, 0.0)
    return {
        "relu": relu,
        "relu_grad": (x > 0).astype(np.float64),
        "sigmoid": sigmoid,
        "sigmoid_grad": sigmoid * (1 - sigmoid),
        "tanh": tanh,
        "tanh_grad": 1 - tanh * tanh,
    }


def test_relu_sigmoid_tanh():
    out = relu_sigmoid_tanh(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(out["relu"], [0, 0, 1])
    assert np.allclose(out["relu_grad"], [0, 0, 1])
    assert np.allclose(out["sigmoid"][1], 0.5)
    assert np.allclose(out["tanh"][1], 0.0)

test_relu_sigmoid_tanh()
print("All tests passed.")
