"""Lightweight tests for pca_first_component.

Run with: python tests.py
"""

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


def test_pca_first_component():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    comp = pca_first_component(X)
    assert np.allclose(comp, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    assert np.allclose(np.linalg.norm(comp), 1.0)

test_pca_first_component()
print("All tests passed.")
