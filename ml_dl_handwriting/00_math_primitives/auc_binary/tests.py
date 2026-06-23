"""Lightweight tests for auc_binary.

Run with: python tests.py
"""

import numpy as np

def auc_binary(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires at least one positive and one negative sample")
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return total / (len(pos) * len(neg))


def test_auc_binary():
    assert np.allclose(auc_binary([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
    assert np.allclose(auc_binary([0, 1], [0.5, 0.5]), 0.5)
    try:
        auc_binary([1, 1], [0.2, 0.3])
        assert False
    except ValueError:
        pass

test_auc_binary()
print("All tests passed.")
