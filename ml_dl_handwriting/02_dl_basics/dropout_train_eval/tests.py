"""Lightweight tests for dropout_train_eval.

Run with: python tests.py
"""

import numpy as np

def dropout_train_eval(X, p=0.5, training=True, seed=None):
    X = np.asarray(X, dtype=np.float64)
    if not training:
        return X
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - p
    mask = (rng.random(X.shape) < keep_prob).astype(np.float64)
    return X * mask / keep_prob


def test_dropout_train_eval():
    X = np.ones(1000)
    out = dropout_train_eval(X, p=0.2, training=True, seed=0)
    assert set(np.unique(out)).issubset({0.0, 1.25})
    assert abs(out.mean() - 1.0) < 0.1
    assert np.allclose(dropout_train_eval(X, p=0.2, training=False), X)

test_dropout_train_eval()
print("All tests passed.")
