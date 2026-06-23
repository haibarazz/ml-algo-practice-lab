"""Lightweight tests for euclidean_distance_matrix.

Run with: python tests.py
"""

import numpy as np

def euclidean_distance_matrix(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def test_euclidean_distance_matrix():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[0.0, 0.0], [6.0, 8.0]])
    D = euclidean_distance_matrix(X, Y)
    assert D.shape == (2, 2)
    assert np.allclose(D, np.array([[0.0, 10.0], [5.0, 5.0]]))
    assert np.allclose(euclidean_distance_matrix(X, X).diagonal(), np.zeros(2))

test_euclidean_distance_matrix()
print("All tests passed.")
