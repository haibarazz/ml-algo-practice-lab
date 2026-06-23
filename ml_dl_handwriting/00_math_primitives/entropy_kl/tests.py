"""Lightweight tests for entropy_kl.

Run with: python tests.py
"""

import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    if q is None:
        return entropy
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sum(q)
    q_safe = np.clip(q, eps, 1.0)
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))
    return entropy, kl


def test_entropy_kl():
    assert np.allclose(entropy_kl([0.5, 0.5]), np.log(2))
    entropy, kl = entropy_kl([0.5, 0.5], [0.25, 0.75])
    assert np.allclose(entropy, np.log(2))
    assert kl > 0
    assert np.allclose(entropy_kl([1.0, 0.0]), 0.0, atol=1e-8)

test_entropy_kl()
print("All tests passed.")
