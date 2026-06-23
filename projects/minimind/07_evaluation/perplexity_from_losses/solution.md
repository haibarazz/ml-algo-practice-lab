# Perplexity from Losses Solution

## Reference Implementation

```python
import math

def perplexity_from_losses(losses, token_counts):
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise ValueError("token_counts must sum to a positive value")
    mean_loss = sum(l * n for l, n in zip(losses, token_counts)) / total_tokens
    return math.exp(mean_loss)
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
