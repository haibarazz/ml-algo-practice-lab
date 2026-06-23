# Euclidean Distance Matrix Solution

## Reference Implementation

```python
import numpy as np

def euclidean_distance_matrix(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
```

## Explanation

1. `X[:, None, :]` 把 X 扩成 `[m, 1, d]`。
2. `Y[None, :, :]` 把 Y 扩成 `[1, n, d]`。
3. 广播相减后沿特征维求平方和，再开方。
4. 实际 KNN / KMeans 中常用平方距离避免开方。
