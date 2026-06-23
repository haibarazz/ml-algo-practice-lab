"""Lightweight tests for dpo_loss.

Run with: python tests.py
"""

import numpy as np

def _logsigmoid(x):
    return -np.logaddexp(0.0, -x)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    policy_chosen = np.asarray(policy_chosen, dtype=np.float64)
    policy_rejected = np.asarray(policy_rejected, dtype=np.float64)
    ref_chosen = np.asarray(ref_chosen, dtype=np.float64)
    ref_rejected = np.asarray(ref_rejected, dtype=np.float64)
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return np.mean(-_logsigmoid(logits))


def test_dpo_loss():
    loss_good = dpo_loss([3.0], [1.0], [2.0], [1.5], beta=1.0)
    loss_bad = dpo_loss([1.0], [3.0], [2.0], [1.5], beta=1.0)
    assert loss_good < loss_bad
    assert np.isfinite(dpo_loss([1000.0], [0.0], [0.0], [0.0]))

test_dpo_loss()
print("All tests passed.")
