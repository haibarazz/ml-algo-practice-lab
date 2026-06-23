"""Lightweight tests for ffn_relu_gelu.

Run with: python tests.py
"""

import numpy as np

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
    hidden = X @ W1 + b1
    if activation == 'relu':
        hidden = np.maximum(hidden, 0.0)
    elif activation == 'gelu':
        hidden = _gelu(hidden)
    else:
        raise ValueError("activation must be relu or gelu")
    return hidden @ W2 + b2


def test_ffn_relu_gelu():
    X = np.array([[1.0, -1.0]])
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.ones((2, 1))
    b2 = np.array([0.0])
    assert np.allclose(ffn_relu_gelu(X, W1, b1, W2, b2, 'relu'), np.array([[1.0]]))
    out = ffn_relu_gelu(X, W1, b1, W2, b2, 'gelu')
    assert out.shape == (1, 1)

test_ffn_relu_gelu()
print("All tests passed.")
