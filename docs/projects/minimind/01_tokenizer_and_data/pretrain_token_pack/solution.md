# Pretrain Token Pack Solution

## Reference Implementation

```python
def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
    content_len = max_length - 2
    content = list(tokens)[:content_len]
    input_ids = [bos_id] + content + [eos_id]
    input_ids = input_ids + [pad_id] * (max_length - len(input_ids))
    labels = [tok if tok != pad_id else -100 for tok in input_ids]
    return input_ids, labels
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
