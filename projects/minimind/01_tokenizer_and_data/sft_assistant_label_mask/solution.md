# SFT Assistant Label Mask Solution

## Reference Implementation

```python
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
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
