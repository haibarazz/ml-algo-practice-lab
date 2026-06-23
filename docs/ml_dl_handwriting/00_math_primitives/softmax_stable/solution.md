# Softmax Stable Solution

## Reference Implementation

```python
import numpy as np


def softmax_stable(x, axis=-1):
    """Compute numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

## Explanation

1. `np.max(..., keepdims=True)` 保留维度，后续广播不会错。
2. `shifted = x - max(x)` 避免 `exp(1000)` 这类溢出。
3. softmax 的输出应该沿指定维度求和为 1。
4. 输入可以是一维 logits，也可以是二维 batch logits。
