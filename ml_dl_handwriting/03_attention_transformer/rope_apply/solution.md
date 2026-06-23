# RoPE Apply Solution

## Reference Implementation

```python
import numpy as np

def rope_apply(x, base=10000):
    x = np.asarray(x, dtype=np.float64)
    seq_len, dim = x.shape
    assert dim % 2 == 0
    pos = np.arange(seq_len)[:, None]
    freq = 1.0 / (base ** (np.arange(0, dim, 2)[None, :] / dim))
    theta = pos * freq
    cos = np.cos(theta)
    sin = np.sin(theta)
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out
```

## Explanation

1. RoPE 要求旋转维度为偶数。
2. pos=0 旋转角为 0，输出不变。
3. 旋转保持每个二维 pair 的范数。
