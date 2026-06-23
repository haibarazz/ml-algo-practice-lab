"""Lightweight tests for kv_cache_step_slice.

Run with: python tests.py
"""

def slice_new_tokens(input_ids, past_len):
    return [list(row[past_len:]) for row in input_ids]


def test_slice_new_tokens():
    ids = [[1, 5, 6], [1, 7, 8]]
    assert slice_new_tokens(ids, 0) == ids
    assert slice_new_tokens(ids, 2) == [[6], [8]]


test_slice_new_tokens()
print("All tests passed.")
