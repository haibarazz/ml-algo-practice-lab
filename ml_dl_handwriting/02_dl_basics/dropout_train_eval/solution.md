# Dropout Train Eval Solution

## Reference Implementation

```python
import numpy as np

def dropout_train_eval(X, p=0.5, training=True, seed=None):
    X = np.asarray(X, dtype=np.float64)
    if not training:
        return X
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - p
    mask = (rng.random(X.shape) < keep_prob).astype(np.float64)
    return X * mask / keep_prob
```

## Explanation

1. 训练时用随机 mask。
2. 除以 keep_prob 保持期望不变。
3. 推理时不做随机丢弃。
