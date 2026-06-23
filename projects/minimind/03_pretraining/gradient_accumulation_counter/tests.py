"""Lightweight tests for gradient_accumulation_counter.

Run with: python tests.py
"""

def accumulation_plan(num_batches, accumulation_steps):
    updates = []
    for step in range(1, num_batches + 1):
        if step % accumulation_steps == 0:
            updates.append(step)
    if num_batches > 0 and num_batches % accumulation_steps != 0:
        updates.append(num_batches)
    return updates


def test_accumulation_plan():
    assert accumulation_plan(8, 4) == [4, 8]
    assert accumulation_plan(10, 4) == [4, 8, 10]
    assert accumulation_plan(3, 8) == [3]


test_accumulation_plan()
print("All tests passed.")
