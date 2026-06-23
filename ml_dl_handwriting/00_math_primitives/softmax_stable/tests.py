"""Lightweight tests for softmax_stable.

Run with: python tests.py
"""

import numpy as np


def softmax_stable(x, axis=-1):
    """Compute numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def test_softmax_stable():
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax_stable(logits)
    assert np.allclose(probs.sum(), 1.0)
    assert np.argmax(probs) == 2

    large_logits = np.array([1000.0, 1001.0, 1002.0])
    large_probs = softmax_stable(large_logits)
    expected = softmax_stable(np.array([0.0, 1.0, 2.0]))
    assert np.allclose(large_probs, expected)
    assert np.all(np.isfinite(large_probs))

    batch = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    batch_probs = softmax_stable(batch, axis=1)
    assert np.allclose(batch_probs.sum(axis=1), np.ones(2))
    assert np.allclose(batch_probs[1], np.array([1 / 3, 1 / 3, 1 / 3]))


test_softmax_stable()
print("All tests passed.")
