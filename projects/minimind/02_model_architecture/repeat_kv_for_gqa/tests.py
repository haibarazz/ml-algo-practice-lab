"""Lightweight tests for repeat_kv_for_gqa.

Run with: python tests.py
"""

import numpy as np

def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return np.repeat(x[:, :, :, None, :], n_rep, axis=3).reshape(b, s, h * n_rep, d)


import numpy as np


def test_repeat_kv():
    x = np.array([[[[1], [2]], [[3], [4]]]])
    out = repeat_kv(x, 2)
    assert out.shape == (1, 2, 4, 1)
    assert out[0, 0, :, 0].tolist() == [1, 1, 2, 2]
    assert repeat_kv(x, 1).shape == x.shape


test_repeat_kv()
print("All tests passed.")
