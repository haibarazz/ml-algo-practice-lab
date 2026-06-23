# DPO Pair Shift Solution

## Reference Implementation

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    return {
        "x_chosen": list(chosen_ids[:-1]),
        "y_chosen": list(chosen_ids[1:]),
        "mask_chosen": list(chosen_mask[1:]),
        "x_rejected": list(rejected_ids[:-1]),
        "y_rejected": list(rejected_ids[1:]),
        "mask_rejected": list(rejected_mask[1:]),
    }
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
