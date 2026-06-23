# Empty Think Cleanup Solution

## Reference Implementation

```python
def cleanup_empty_think(prompt, keep_empty=False):
    pattern = "<think>

</think>

"
    if keep_empty:
        return prompt
    return prompt.replace(pattern, "")
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
