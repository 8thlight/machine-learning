"""Tests for the .padding file"""
import numpy as np

from ..padding import padding


def test_padding():
    """Adds 0 padding of size 1 to the img"""
    img = np.array([
        [1, 1],
        [1, 1]
    ])
    output = padding(img, 1)

    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ])

    assert (output == expected).all()


def test_padding_2():
    """Adds 0 padding of size 2 to the img"""
    img = np.array([
        [1, 1],
        [1, 1]
    ])
    output = padding(img, 2)

    expected = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    assert (output == expected).all()


def test_padding_2_chan():
    """Padding a 3D img results in a 3D output"""
    img = np.array([
        [
            [1, 1],
            [1, 1]
        ],
        [
            [2, 2],
            [2, 2]
        ]
    ])
    output = padding(img, 1)
    expected = np.array([
        padding(img[0], 1),
        padding(img[1], 1)
    ])

    assert (output == expected).all()
