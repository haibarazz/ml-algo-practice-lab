"""Lightweight tests for sft_label_mask.

Run with: python tests.py
"""

import numpy as np

def sft_label_mask(input_ids, prompt_lengths, pad_token_id=0, ignore_index=-100):
    input_ids = np.asarray(input_ids, dtype=np.int64)
    labels = input_ids.copy()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = ignore_index
    labels[input_ids == pad_token_id] = ignore_index
    return labels


def test_sft_label_mask():
    ids = np.array([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    labels = sft_label_mask(ids, [2, 1], pad_token_id=0)
    assert labels.tolist() == [[-100, -100, 3, 4, -100], [-100, 6, 7, -100, -100]]

test_sft_label_mask()
print("All tests passed.")
