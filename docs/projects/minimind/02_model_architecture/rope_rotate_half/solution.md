# RoPE Rotate Half Solution

## Reference Implementation

```python
import numpy as np

def apply_rope_rotate_half(x, cos, sin):
    x = np.asarray(x, dtype=np.float64)
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.asarray(cos) + rotated * np.asarray(sin)
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
