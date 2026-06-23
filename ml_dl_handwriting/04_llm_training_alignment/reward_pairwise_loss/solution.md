# Reward Pairwise Loss Solution

## Reference Implementation

```python
import numpy as np

def reward_pairwise_loss(chosen_rewards, rejected_rewards):
    chosen_rewards = np.asarray(chosen_rewards, dtype=np.float64)
    rejected_rewards = np.asarray(rejected_rewards, dtype=np.float64)
    diff = chosen_rewards - rejected_rewards
    return np.mean(np.logaddexp(0.0, -diff))
```

## Explanation

1. 只关心 chosen 和 rejected 的差。
2. 差越大，loss 越小。
3. `logaddexp(0, -diff)` 是 `-logsigmoid(diff)` 的稳定写法。
