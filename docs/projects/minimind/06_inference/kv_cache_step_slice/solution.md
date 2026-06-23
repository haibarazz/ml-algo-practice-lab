# KV Cache Step Slice Solution

## Reference Implementation

```python
def slice_new_tokens(input_ids, past_len):
    return [list(row[past_len:]) for row in input_ids]
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
