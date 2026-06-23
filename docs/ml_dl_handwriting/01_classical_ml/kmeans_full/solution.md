# KMeans Full Solution

## Reference Implementation

```python
import numpy as np

def kmeans_full(points, initial_centers, max_iters=100, tol=1e-6):
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(initial_centers, dtype=np.float64).copy()
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(centers.shape[0]):
            assigned = points[labels == k]
            if len(assigned) > 0:
                new_centers[k] = assigned.mean(axis=0)
        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers
    return labels, centers
```

## Explanation

1. 每轮先重新分配标签。
2. 再用每个簇的样本均值更新中心。
3. 中心变化小于 `tol` 即停止。
4. 空簇保留旧中心，避免 NaN。
