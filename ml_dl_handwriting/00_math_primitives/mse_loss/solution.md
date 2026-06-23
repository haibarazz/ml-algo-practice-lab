# MSE Loss Solution

## Reference Implementation

```python
import numpy as np

def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    return np.mean(diff * diff)
```

## Explanation

1. 先统一转换为浮点数组。
2. 按所有元素求平均，而不是只按 batch 维。
3. 如果后续写 backward，梯度分母要和这里的 mean 定义一致。
