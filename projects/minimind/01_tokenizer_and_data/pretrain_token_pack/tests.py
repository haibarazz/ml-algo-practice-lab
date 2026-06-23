"""Lightweight tests for pretrain_token_pack.

Run with: python tests.py
"""

def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
    content_len = max_length - 2
    content = list(tokens)[:content_len]
    input_ids = [bos_id] + content + [eos_id]
    input_ids = input_ids + [pad_id] * (max_length - len(input_ids))
    labels = [tok if tok != pad_id else -100 for tok in input_ids]
    return input_ids, labels


def test_pack_pretrain_tokens():
    input_ids, labels = pack_pretrain_tokens([7, 8, 9, 10], bos_id=1, eos_id=2, pad_id=0, max_length=5)
    assert input_ids == [1, 7, 8, 9, 2]
    assert labels == [1, 7, 8, 9, 2]
    input_ids, labels = pack_pretrain_tokens([7], bos_id=1, eos_id=2, pad_id=0, max_length=5)
    assert input_ids == [1, 7, 2, 0, 0]
    assert labels == [1, 7, 2, -100, -100]


test_pack_pretrain_tokens()
print("All tests passed.")
