"""Lightweight tests for kmeans_full.

Run with: python tests.py
"""

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


def test_kmeans_full():
    points = np.array([[0, 0], [0, 2], [10, 10], [10, 12]], dtype=float)
    labels, centers = kmeans_full(points, np.array([[0, 0], [10, 10]], dtype=float))
    assert labels.tolist() == [0, 0, 1, 1]
    assert np.allclose(centers, np.array([[0, 1], [10, 11]], dtype=float))

test_kmeans_full()
print("All tests passed.")
