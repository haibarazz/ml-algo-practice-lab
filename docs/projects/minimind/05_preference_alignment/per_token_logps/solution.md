# Per Token Log Probs Solution

## Reference Implementation

```python
import numpy as np

def per_token_logps(logits, token_ids):
    logits = np.asarray(logits, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    batch = np.arange(token_ids.shape[0])[:, None]
    pos = np.arange(token_ids.shape[1])[None, :]
    return log_probs[batch, pos, token_ids]
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
