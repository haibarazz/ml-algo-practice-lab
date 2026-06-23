"""Lightweight tests for dpo_sequence_loss.

Run with: python tests.py
"""

import numpy as np

def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (np.asarray(ref_log_probs) * np.asarray(mask)).sum(axis=1)
    policy_log_probs = (np.asarray(policy_log_probs) * np.asarray(mask)).sum(axis=1)
    half = len(ref_log_probs) // 2
    chosen_ref, reject_ref = ref_log_probs[:half], ref_log_probs[half:]
    chosen_pol, reject_pol = policy_log_probs[:half], policy_log_probs[half:]
    logits = (chosen_pol - reject_pol) - (chosen_ref - reject_ref)
    return float(np.mean(np.logaddexp(0.0, -beta * logits)))


import numpy as np


def test_dpo_sequence_loss():
    ref = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    pol_good = np.array([[-0.1, -0.1], [-2.0, -2.0]])
    pol_bad = np.array([[-2.0, -2.0], [-0.1, -0.1]])
    mask = np.ones_like(ref)
    assert dpo_sequence_loss(ref, pol_good, mask, beta=1.0) < dpo_sequence_loss(ref, pol_bad, mask, beta=1.0)


test_dpo_sequence_loss()
print("All tests passed.")
