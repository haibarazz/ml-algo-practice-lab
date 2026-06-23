"""Lightweight tests for preference_pair_format.

Run with: python tests.py
"""

def preference_pair_format(prompts, chosen, rejected):
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, rejected must have the same length")
    pairs = []
    for p, c, r in zip(prompts, chosen, rejected):
        pairs.append({"prompt": p, "chosen": c, "rejected": r})
    return pairs


def test_preference_pair_format():
    pairs = preference_pair_format(["Q1", "Q2"], ["good1", "good2"], ["bad1", "bad2"])
    assert pairs[0] == {"prompt": "Q1", "chosen": "good1", "rejected": "bad1"}
    try:
        preference_pair_format(["Q"], ["A"], [])
        assert False
    except ValueError:
        pass

test_preference_pair_format()
print("All tests passed.")
