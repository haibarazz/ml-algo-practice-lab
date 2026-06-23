"""Lightweight tests for causal_lm_shift_loss.

Run with: python tests.py
"""

import numpy as np

def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    x = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    y = labels[:, 1:].reshape(-1)
    mask = y != ignore_index
    x = x[mask]
    y = y[mask]
    shifted = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    correct = shifted[np.arange(len(y)), y]
    return float(np.mean(-correct + logsumexp))


import numpy as np


def test_causal_lm_shift_loss():
    logits = np.array([[[0, 5, 0], [0, 0, 5], [5, 0, 0]]], dtype=float)
    labels = np.array([[0, 1, 2]])
    assert causal_lm_shift_loss(logits, labels) < 0.02
    masked = np.array([[0, -100, 2]])
    assert np.isfinite(causal_lm_shift_loss(logits, masked))


test_causal_lm_shift_loss()
print("All tests passed.")
