"""Lightweight tests for knn_classifier.

Run with: python tests.py
"""

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


def test_knn_classifier():
    X = np.array([[0, 0], [0, 1], [5, 5], [6, 5]])
    y = np.array([0, 0, 1, 1])
    q = np.array([[0.2, 0.1], [5.5, 5.0]])
    assert knn_classifier(X, y, q, k=3).tolist() == [0, 1]
    assert knn_classifier([[0], [2]], [1, 0], [[1]], k=2).tolist() == [0]

test_knn_classifier()
print("All tests passed.")
