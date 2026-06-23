"""Lightweight tests for cosine_lr_schedule.

Run with: python tests.py
"""

import math

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


import math


def test_get_lr():
    assert abs(get_lr(0, 100, 1.0) - 1.0) < 1e-12
    assert abs(get_lr(100, 100, 1.0) - 0.1) < 1e-12
    assert abs(get_lr(50, 100, 2.0) - 1.1) < 1e-12


test_get_lr()
print("All tests passed.")
