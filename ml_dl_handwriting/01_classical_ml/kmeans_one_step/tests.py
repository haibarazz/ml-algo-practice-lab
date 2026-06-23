"""Lightweight tests for kmeans_one_step.

Run with: python tests.py
"""

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


def test_kmeans_one_step():
    points = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [10.0, 10.0],
        [10.0, 12.0],
    ])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, new_centers = kmeans_one_step(points, centers)
    assert labels.tolist() == [0, 0, 1, 1]
    assert np.allclose(new_centers, np.array([[0.0, 1.0], [10.0, 11.0]]))

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, new_centers = kmeans_one_step(points, centers)
    assert labels.tolist() == [0, 0]
    assert np.allclose(new_centers[0], np.array([0.5, 0.0]))
    assert np.allclose(new_centers[1], centers[1])  # empty cluster keeps old center


test_kmeans_one_step()
print("All tests passed.")
