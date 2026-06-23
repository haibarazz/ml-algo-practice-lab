"""Lightweight tests for sft_assistant_label_mask.

Run with: python tests.py
"""

def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(assistant_bos)] == assistant_bos:
            start = i + len(assistant_bos)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(assistant_eos)] == assistant_eos:
                    break
                end += 1
            stop = min(end + len(assistant_eos), max_length)
            for j in range(start, stop):
                labels[j] = input_ids[j]
            i = stop
        else:
            i += 1
    return labels


def test_generate_assistant_labels():
    ids = [9, 101, 102, 5, 6, 201, 7, 101, 102, 8, 201, 0]
    labels = generate_assistant_labels(ids, assistant_bos=[101, 102], assistant_eos=[201], max_length=len(ids))
    assert labels == [-100, -100, -100, 5, 6, 201, -100, -100, -100, 8, 201, -100]


test_generate_assistant_labels()
print("All tests passed.")
