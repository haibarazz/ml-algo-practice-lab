# RMSNorm Forward Solution

## Reference Implementation

```python
import numpy as np

def rms_norm_forward(X, gamma, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)
    rms = np.sqrt(np.mean(X * X, axis=-1, keepdims=True) + eps)
    return X / rms * gamma
```

## Explanation

1. RMS 是平方均值再开方。
2. 不减均值，因此输出均值不一定为 0。
3. LLaMA 等模型常用 RMSNorm 变体。
