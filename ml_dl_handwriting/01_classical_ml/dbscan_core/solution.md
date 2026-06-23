# DBSCAN Core Solution

## Reference Implementation

```python
import numpy as np

def dbscan_core(points, eps, min_samples):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(distances[i] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.where(distances[j] <= eps)[0].tolist()
                if len(j_neighbors) >= min_samples:
                    for nb in j_neighbors:
                        if nb not in queue:
                            queue.append(nb)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels
```

## Explanation

1. 先预计算两两距离矩阵，便于教学。
2. 遍历未访问点，判断是否核心点。
3. 核心点启动 BFS 扩展簇。
4. 噪声点可能后来被某个簇吸收。
