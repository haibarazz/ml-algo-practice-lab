"""Lightweight tests for adam_optimizer.

Run with: python tests.py
"""

import numpy as np

def adam_optimizer(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_param, m, v


def test_adam_optimizer():
    p = np.array([1.0, 2.0])
    g = np.array([0.1, -0.2])
    m = np.zeros(2); v = np.zeros(2)
    new_p, m, v = adam_optimizer(p, g, m, v, t=1, lr=0.001)
    assert new_p[0] < p[0]
    assert new_p[1] > p[1]
    assert np.all(m != 0)
    assert np.all(v > 0)

test_adam_optimizer()
print("All tests passed.")
