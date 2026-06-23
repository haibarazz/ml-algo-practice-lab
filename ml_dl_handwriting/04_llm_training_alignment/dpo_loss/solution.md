# DPO Loss Solution

## Reference Implementation

```python
import numpy as np

def _logsigmoid(x):
    return -np.logaddexp(0.0, -x)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    policy_chosen = np.asarray(policy_chosen, dtype=np.float64)
    policy_rejected = np.asarray(policy_rejected, dtype=np.float64)
    ref_chosen = np.asarray(ref_chosen, dtype=np.float64)
    ref_rejected = np.asarray(ref_rejected, dtype=np.float64)
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return np.mean(-_logsigmoid(logits))
```

## Explanation

1. 先算 policy preference margin。
2. 再减 reference preference margin。
3. 乘 beta 控制偏好强度。
4. 用稳定 logsigmoid。
