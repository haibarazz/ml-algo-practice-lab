"""Lightweight tests for moe_router_aux_loss.

Run with: python tests.py
"""

import numpy as np

def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    scores = np.asarray(scores, dtype=np.float64)
    topk_idx = np.asarray(topk_idx, dtype=np.int64)
    one_hot = np.eye(num_experts)[topk_idx]
    load = one_hot.mean(axis=0)
    mean_score = scores.mean(axis=0)
    return float((load * mean_score).sum() * num_experts * coef)


import numpy as np


def test_moe_router_aux_loss():
    scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    topk_idx = np.array([[0], [1]])
    loss = moe_router_aux_loss(scores, topk_idx, num_experts=3, coef=0.01)
    assert np.allclose(loss, 0.0135)


test_moe_router_aux_loss()
print("All tests passed.")
