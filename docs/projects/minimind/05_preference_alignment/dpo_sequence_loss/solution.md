# DPO Sequence Loss Solution

## Reference Implementation

```python
import numpy as np

def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (np.asarray(ref_log_probs) * np.asarray(mask)).sum(axis=1)
    policy_log_probs = (np.asarray(policy_log_probs) * np.asarray(mask)).sum(axis=1)
    half = len(ref_log_probs) // 2
    chosen_ref, reject_ref = ref_log_probs[:half], ref_log_probs[half:]
    chosen_pol, reject_pol = policy_log_probs[:half], policy_log_probs[half:]
    logits = (chosen_pol - reject_pol) - (chosen_ref - reject_ref)
    return float(np.mean(np.logaddexp(0.0, -beta * logits)))
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
