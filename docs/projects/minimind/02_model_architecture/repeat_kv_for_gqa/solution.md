# Repeat KV for GQA Solution

## Reference Implementation

```python
import numpy as np

def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return np.repeat(x[:, :, :, None, :], n_rep, axis=3).reshape(b, s, h * n_rep, d)
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。
