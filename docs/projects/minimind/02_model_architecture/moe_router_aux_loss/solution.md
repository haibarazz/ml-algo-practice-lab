# MoE Router Aux Loss Solution

## Reference Implementation

```python
import numpy as np

def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    scores = np.asarray(scores, dtype=np.float64)
    topk_idx = np.asarray(topk_idx, dtype=np.int64)
    one_hot = np.eye(num_experts)[topk_idx]
    load = one_hot.mean(axis=0)
    mean_score = scores.mean(axis=0)
    return float((load * mean_score).sum() * num_experts * coef)
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
