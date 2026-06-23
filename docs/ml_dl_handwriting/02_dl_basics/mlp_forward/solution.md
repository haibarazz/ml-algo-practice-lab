# MLP Forward Solution

## Reference Implementation

```python
import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    X = np.asarray(X, dtype=np.float64)
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    out = a1 @ W2 + b2
    return out, {"z1": z1, "a1": a1}
```

## Explanation

1. 第一层线性变换后接 ReLU。
2. 第二层线性输出预测。
3. 返回 cache 是为了后续 backward 使用。
