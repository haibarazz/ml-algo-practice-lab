# Preference Pair Format Solution

## Reference Implementation

```python
def preference_pair_format(prompts, chosen, rejected):
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, rejected must have the same length")
    pairs = []
    for p, c, r in zip(prompts, chosen, rejected):
        pairs.append({"prompt": p, "chosen": c, "rejected": r})
    return pairs
```

## Explanation

1. 三个列表长度必须一致。
2. 每条样本保留 prompt/chosen/rejected 三个字段。
3. 这是后续 DPO/RM 的最小数据单元。
