"""Lightweight tests for contrastive_loss.

Run with: python tests.py
"""

import numpy as np

def _normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def contrastive_loss(query, positive, negatives, temperature=1.0):
    q = _normalize(np.asarray(query, dtype=np.float64))
    p = _normalize(np.asarray(positive, dtype=np.float64))
    n = _normalize(np.asarray(negatives, dtype=np.float64))
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum('bd,bkd->bk', q, n)
    logits = np.concatenate([pos, neg], axis=1) / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return np.mean(-log_probs[:, 0])


def test_contrastive_loss():
    q = np.array([[1.0, 0.0]])
    p = np.array([[1.0, 0.0]])
    neg = np.array([[[0.0, 1.0], [-1.0, 0.0]]])
    good = contrastive_loss(q, p, neg, temperature=0.5)
    bad = contrastive_loss(q, np.array([[0.0, 1.0]]), neg, temperature=0.5)
    assert good < bad

test_contrastive_loss()
print("All tests passed.")
