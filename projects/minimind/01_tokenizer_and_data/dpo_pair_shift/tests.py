"""Lightweight tests for dpo_pair_shift.

Run with: python tests.py
"""

def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    return {
        "x_chosen": list(chosen_ids[:-1]),
        "y_chosen": list(chosen_ids[1:]),
        "mask_chosen": list(chosen_mask[1:]),
        "x_rejected": list(rejected_ids[:-1]),
        "y_rejected": list(rejected_ids[1:]),
        "mask_rejected": list(rejected_mask[1:]),
    }


def test_build_dpo_shifted():
    out = build_dpo_shifted([1, 4, 5, 2], [1, 4, 6, 2], [0, 1, 1, 1], [0, 1, 1, 1])
    assert out["x_chosen"] == [1, 4, 5]
    assert out["y_chosen"] == [4, 5, 2]
    assert out["mask_chosen"] == [1, 1, 1]
    assert out["x_rejected"] == [1, 4, 6]
    assert out["y_rejected"] == [4, 6, 2]


test_build_dpo_shifted()
print("All tests passed.")
