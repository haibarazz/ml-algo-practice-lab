"""Lightweight tests for per_token_logps.

Run with: python tests.py
"""

import numpy as np

def per_token_logps(logits, token_ids):
    logits = np.asarray(logits, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    batch = np.arange(token_ids.shape[0])[:, None]
    pos = np.arange(token_ids.shape[1])[None, :]
    return log_probs[batch, pos, token_ids]


import numpy as np


def test_per_token_logps():
    logits = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    ids = np.array([[0, 1]])
    out = per_token_logps(logits, ids)
    expected = np.array([[-np.log1p(np.exp(-2.0)), -np.log1p(np.exp(-2.0))]])
    assert np.allclose(out, expected)


test_per_token_logps()
print("All tests passed.")
