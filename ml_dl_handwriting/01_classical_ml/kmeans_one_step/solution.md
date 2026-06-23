# KMeans One Step Solution

## Reference Implementation

```python
import numpy as np


def kmeans_one_step(points, centers):
    """Run one KMeans assignment-update step.

    Returns:
        labels: shape [num_points]
        new_centers: shape [num_centers, dim]
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    diff = points[:, None, :] - centers[None, :, :]
    distances = np.sum(diff * diff, axis=-1)
    labels = np.argmin(distances, axis=1)

    new_centers = centers.copy()
    for k in range(centers.shape[0]):
        assigned = points[labels == k]
        if assigned.shape[0] > 0:
            new_centers[k] = assigned.mean(axis=0)

    return labels, new_centers
```

## Explanation

1. `points[:, None, :] - centers[None, :, :]` 得到 `[N, K, D]` 的广播差值。
2. 沿最后一维求平方和，得到 `[N, K]` 距离矩阵。
3. `argmin(axis=1)` 给每个样本选最近中心。
4. 更新中心时必须处理空簇。这个样板选择保留旧中心，是面试里最容易漏掉的边界。
