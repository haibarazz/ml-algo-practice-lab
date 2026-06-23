"""Lightweight tests for causal_lm_loss.

Run with: python tests.py
"""

import numpy as np

def causal_lm_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    mask = flat_labels != ignore_index
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]
    shifted = flat_logits - np.max(flat_logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(len(flat_labels)), flat_labels]
    return np.mean(-correct + logsumexp)


def test_causal_lm_loss():
    logits = np.array([[[0, 5, 0], [0, 0, 5], [5, 0, 0]]], dtype=float)
    labels = np.array([[0, 1, 2]])
    loss = causal_lm_loss(logits, labels)
    assert loss < 0.02
    labels_masked = np.array([[0, -100, 2]])
    assert np.isfinite(causal_lm_loss(logits, labels_masked))

test_causal_lm_loss()
print("All tests passed.")
