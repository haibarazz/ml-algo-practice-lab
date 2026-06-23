"""Lightweight tests for top_k_top_p_filter.

Run with: python tests.py
"""

import numpy as np

def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64).copy()
    if top_k > 0:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        finite = np.isfinite(sorted_logits)
        shifted = sorted_logits[finite] - np.max(sorted_logits[finite])
        probs = np.exp(shifted) / np.exp(shifted).sum()
        remove = np.zeros_like(sorted_logits, dtype=bool)
        finite_indices = np.where(finite)[0]
        cumulative = np.cumsum(probs)
        remove[finite_indices] = cumulative > top_p
        if len(remove) > 0:
            remove[0] = False
        logits[order[remove]] = -np.inf
    return logits


import numpy as np


def test_top_k_top_p_filter():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    out = top_k_top_p_filter(logits, top_k=2, top_p=1.0)
    assert np.isneginf(out[2]) and np.isneginf(out[3])
    out = top_k_top_p_filter(logits, top_k=0, top_p=0.7)
    assert not np.isneginf(out[0])
    assert np.isneginf(out[-1])


test_top_k_top_p_filter()
print("All tests passed.")
