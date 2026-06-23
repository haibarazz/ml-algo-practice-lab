"""Lightweight tests for em_gmm_one_step.

Run with: python tests.py
"""

import numpy as np

def _normal_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

def em_gmm_one_step(x, weights, means, variances):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    probs = np.stack([weights[k] * _normal_pdf(x, means[k], variances[k]) for k in range(len(weights))], axis=1)
    resp = probs / probs.sum(axis=1, keepdims=True)
    Nk = resp.sum(axis=0)
    new_weights = Nk / len(x)
    new_means = (resp * x[:, None]).sum(axis=0) / Nk
    new_vars = (resp * (x[:, None] - new_means[None, :]) ** 2).sum(axis=0) / Nk
    return resp, new_weights, new_means, new_vars


def test_em_gmm_one_step():
    x = np.array([-1.0, 0.0, 5.0, 6.0])
    resp, w, m, v = em_gmm_one_step(x, np.array([0.5, 0.5]), np.array([0.0, 5.0]), np.array([1.0, 1.0]))
    assert resp.shape == (4, 2)
    assert np.allclose(resp.sum(axis=1), np.ones(4))
    assert m[0] < m[1]
    assert np.allclose(w.sum(), 1.0)

test_em_gmm_one_step()
print("All tests passed.")
