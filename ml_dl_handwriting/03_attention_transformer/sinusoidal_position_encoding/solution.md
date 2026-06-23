# Sinusoidal Position Encoding Solution

## Reference Implementation

```python
import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / (10000 ** (i / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    return pe
```

## Explanation

1. 偶数维用 sin，奇数维用 cos。
2. pos=0 时 sin 为 0，cos 为 1。
3. 不同频率覆盖不同尺度的位置变化。
