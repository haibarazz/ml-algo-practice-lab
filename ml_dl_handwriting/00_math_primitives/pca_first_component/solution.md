# PCA First Component Solution

## Reference Implementation

```python
import numpy as np

def pca_first_component(X):
    X = np.asarray(X, dtype=np.float64)
    centered = X - X.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / X.shape[0]
    values, vectors = np.linalg.eigh(cov)
    component = vectors[:, np.argmax(values)]
    component = component / np.linalg.norm(component)
    first_nonzero = np.flatnonzero(np.abs(component) > 1e-12)
    if len(first_nonzero) and component[first_nonzero[0]] < 0:
        component = -component
    return component
```

## Explanation

1. PCA 必须先中心化。
2. 协方差矩阵是对称矩阵，用 `eigh` 更合适。
3. 特征向量正负号不唯一，测试时要固定一个符号约定。
