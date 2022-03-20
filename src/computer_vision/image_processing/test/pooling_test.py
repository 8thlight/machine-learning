"""Tests for the .pooling file"""
import numpy as np

from ..pooling import maxpool, minpool


def test_minpool():
    """Minpool keeps the minimum values of the img according to the area"""
    img = np.array([
        [0, 3, 4, -1],
        [6, 7, 4, 5],
        [9, 7, 6, 4],
        [-2, 8, 4, -1]
    ])
    output = minpool(img, 3)
    expected = np.array([
        [0, -1],
        [-2, -1]
    ])

    assert (output == expected).all()


def test_maxpool():
    """Maxpool keeps the maximum values of the img according to the area"""
    img = np.array([
        [9, 0, 9],
        [1, 1, 1],
        [8, 2, 6]
    ])
    output = maxpool(img, 2)
    expected = np.array([
        [9, 9],
        [8, 6]
    ])

    assert (output == expected).all()
