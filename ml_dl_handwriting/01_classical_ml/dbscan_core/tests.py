"""Lightweight tests for dbscan_core.

Run with: python tests.py
"""

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


def test_dbscan_core():
    points = np.array([[0, 0], [0, 0.1], [5, 5], [5.1, 5], [10, 10]], dtype=float)
    labels = dbscan_core(points, eps=0.25, min_samples=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] == -1

test_dbscan_core()
print("All tests passed.")
