"""Lightweight tests for mlp_backward.

Run with: python tests.py
"""

import numpy as np


def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Forward and backward for a two-layer MLP with ReLU and MSE.

    Returns:
        loss: scalar float
        grads: dict with dW1, db1, dW2, db2
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    pred = a1 @ W2 + b2

    diff = pred - y
    loss = np.mean(diff * diff)

    dpred = 2.0 * diff / diff.size
    dW2 = a1.T @ dpred
    db2 = dpred.sum(axis=0)
    da1 = dpred @ W2.T

    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return loss, grads


def test_mlp_forward_backward():
    X = np.array([[1.0, -1.0], [0.5, 2.0]])
    y = np.array([[1.0], [0.0]])
    W1 = np.array([[0.2, -0.4, 0.1], [0.7, 0.3, -0.5]])
    b1 = np.array([0.0, 0.1, -0.2])
    W2 = np.array([[0.6], [-0.1], [0.2]])
    b2 = np.array([0.05])

    loss, grads = mlp_forward_backward(X, y, W1, b1, W2, b2)
    assert np.isscalar(loss) or np.asarray(loss).shape == ()
    assert set(grads) == {"dW1", "db1", "dW2", "db2"}
    assert grads["dW1"].shape == W1.shape
    assert grads["db1"].shape == b1.shape
    assert grads["dW2"].shape == W2.shape
    assert grads["db2"].shape == b2.shape

    expected_loss = 0.78345
    assert np.allclose(loss, expected_loss, atol=1e-5)
    assert np.allclose(grads["dW2"], np.array([[1.35], [0.45], [-0.348]]), atol=1e-5)
    assert np.allclose(grads["db2"], np.array([0.03]), atol=1e-5)
    assert np.allclose(grads["dW1"], np.array([[0.27, -0.045, -0.174], [1.08, -0.18, 0.174]]), atol=1e-5)
    assert np.allclose(grads["db1"], np.array([0.54, -0.09, -0.174]), atol=1e-5)


test_mlp_forward_backward()
print("All tests passed.")
