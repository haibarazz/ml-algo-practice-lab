"""Lightweight tests for perplexity_from_losses.

Run with: python tests.py
"""

import math

def perplexity_from_losses(losses, token_counts):
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise ValueError("token_counts must sum to a positive value")
    mean_loss = sum(l * n for l, n in zip(losses, token_counts)) / total_tokens
    return math.exp(mean_loss)


import math


def test_perplexity_from_losses():
    ppl = perplexity_from_losses([math.log(2), math.log(4)], [10, 10])
    assert abs(ppl - math.sqrt(8)) < 1e-12
    assert perplexity_from_losses([0.0], [5]) == 1.0


test_perplexity_from_losses()
print("All tests passed.")
