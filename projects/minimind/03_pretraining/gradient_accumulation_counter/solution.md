# Gradient Accumulation Counter Solution

## Reference Implementation

```python
def accumulation_plan(num_batches, accumulation_steps):
    updates = []
    for step in range(1, num_batches + 1):
        if step % accumulation_steps == 0:
            updates.append(step)
    if num_batches > 0 and num_batches % accumulation_steps != 0:
        updates.append(num_batches)
    return updates
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
