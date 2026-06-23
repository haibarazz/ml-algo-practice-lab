"""Lightweight tests for sinusoidal_position_encoding.

Run with: python tests.py
"""

import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / (10000 ** (i / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    return pe


def test_sinusoidal_position_encoding():
    pe = sinusoidal_position_encoding(3, 4)
    assert pe.shape == (3, 4)
    assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert not np.allclose(pe[1], pe[2])

test_sinusoidal_position_encoding()
print("All tests passed.")
