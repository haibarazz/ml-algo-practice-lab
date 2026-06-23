# SwiGLU Feed Forward Solution

## Reference Implementation

```python
import numpy as np

def swiglu_ffn(x, w_gate, w_up, w_down):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ np.asarray(w_gate, dtype=np.float64)
    up = x @ np.asarray(w_up, dtype=np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ np.asarray(w_down, dtype=np.float64)
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
