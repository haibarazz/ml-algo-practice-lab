# SwiGLU Solution

## Reference Implementation

```python
import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def swiglu(X, W_gate, b_gate, W_up, b_up, W_down, b_down):
    gate = X @ W_gate + b_gate
    up = X @ W_up + b_up
    silu_gate = gate * _sigmoid(gate)
    return (silu_gate * up) @ W_down + b_down
```

## Explanation

1. gate 分支经过 SiLU。
2. up 分支保持线性。
3. 两个分支逐元素相乘后 down projection。
4. 这是 LLaMA 系列常见 FFN 结构。
