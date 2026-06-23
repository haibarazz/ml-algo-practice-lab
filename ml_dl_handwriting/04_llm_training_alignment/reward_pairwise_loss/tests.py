"""Lightweight tests for reward_pairwise_loss.

Run with: python tests.py
"""

import numpy as np

def reward_pairwise_loss(chosen_rewards, rejected_rewards):
    chosen_rewards = np.asarray(chosen_rewards, dtype=np.float64)
    rejected_rewards = np.asarray(rejected_rewards, dtype=np.float64)
    diff = chosen_rewards - rejected_rewards
    return np.mean(np.logaddexp(0.0, -diff))


def test_reward_pairwise_loss():
    assert reward_pairwise_loss([3.0], [1.0]) < reward_pairwise_loss([1.0], [3.0])
    loss = reward_pairwise_loss([1.0, 2.0], [0.0, 3.0])
    assert np.isfinite(loss)

test_reward_pairwise_loss()
print("All tests passed.")
