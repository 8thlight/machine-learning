"""Tests for the .convolution file"""
import numpy as np

from ..convolution import convolution, _multiply_sum


def test_multiply_sum():
    """Multiplies two frames element-wise and adds their values into one"""
    img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    kernel = np.array([
        [-1, 1, 2],
        [1, 2, 3],
        [2, 3, -1]
    ])
    output = _multiply_sum(img, kernel)

    assert output == 68


def test_convolution():
    """
    A convolution reduces the img size and is the result of
    element-wise multiplications and additions
    """
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 1, 2],
        [3, 4, 5, 6]
    ])
    kernel = np.array([
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]
    ])

    output = convolution(img, kernel)
    expected = np.array([
        [8, 11],
        [20, 13]
    ])

    assert (output == expected).all()


def test_convolution_2():
    """Second example of convolution"""
    img = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [5, 6, 7, 8],
        [8, 7, 6, 5]
    ])
    kernel = np.array([
        [-1, 1],
        [1, -1]
    ])

    output = convolution(img, kernel)
    expected = np.array([
        [2, 2, 2],
        [-2, -2, -2],
        [2, 2, 2]
    ])

    assert (output == expected).all()


def test_convolution_2_chan():
    """A 3D convolution creates a 2D output"""
    img = np.array([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [3, 4, 5],
            [6, 7, 8],
            [3, 2, 1]
        ]
    ])
    kernel = np.array([
        [
            [-1, 1],
            [-1, 1]
        ],
        [
            [1, -1],
            [1, -1]
        ]
    ])
    output = convolution(img, kernel)

    expected = np.array([
        convolution(img[0], kernel[0]),
        convolution(img[1], kernel[1])
    ]).sum(axis=0)

    assert (output == expected).all()
