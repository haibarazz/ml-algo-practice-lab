"""Lightweight tests for cross_entropy.

Run with: python tests.py
"""

import numpy as np

def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(logits.shape[0]), targets]
    return np.mean(-correct + logsumexp)


def test_cross_entropy():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = np.array([0, 1])
    loss = cross_entropy(logits, targets)
    assert np.allclose(loss, 0.288725992, atol=1e-6)
    large = np.array([[1000.0, 1001.0]])
    assert np.isfinite(cross_entropy(large, np.array([1])))

test_cross_entropy()
print("All tests passed.")
