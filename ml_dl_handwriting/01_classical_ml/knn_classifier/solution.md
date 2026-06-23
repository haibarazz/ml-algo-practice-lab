# KNN Classifier Solution

## Reference Implementation

```python
import numpy as np

def knn_classifier(X_train, y_train, X_query, k=3):
    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    y_train = np.asarray(y_train)
    preds = []
    for q in X_query:
        dist = np.sqrt(np.sum((X_train - q) ** 2, axis=1))
        nearest = np.argsort(dist)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])
    return np.asarray(preds)
```

## Explanation

1. 对每个 query 计算到所有训练样本的距离。
2. `argsort` 取最近 k 个。
3. `np.unique(..., return_counts=True)` 统计投票。
4. `np.unique` 返回有序标签，因此 `argmax` 在票数相同会选择较小标签。
