"""Tests for the .reshaping file"""
import numpy as np

from ..reshaping import channel_as_first_dimension, channel_as_last_dimension


def test_channel_as_first_dimension():
    """Shows the result of extracting RGB values into separate channels"""
    img = np.array([
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    ])
    output = channel_as_first_dimension(img)

    expected = np.array([
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]
        ],
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3]
        ],
    ])

    assert (output == expected).all()


def test_channel_as_last_dimension():
    """Shows the result of grouping separate channels into RGB values"""
    img = np.array([
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]
        ],
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3]
        ],
    ])
    output = channel_as_last_dimension(img)

    expected = np.array([
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    ])

    assert (output == expected).all()


def test_channel_as_first_dimension_same_shape():
    """Order is preserved when the dimension lengths are the same: (3x3x3)"""
    img = np.array([
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    ])
    output = channel_as_first_dimension(img)

    expected = np.array([
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ],
        [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ]
    ])

    assert (output == expected).all()


def test_channel_as_last_dimension_same_shape():
    """Order is preserved when the dimension lengths are the same: (3x3x3)"""
    img = np.array([
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ],
        [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ]
    ])
    output = channel_as_last_dimension(img)

    expected = np.array([
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    ])

    assert (output == expected).all()


def test_symmetry_start_channels_last_dimension():
    """
    Reshaping from RGB into convolution shape and back results in the same image
    """
    img = np.array([
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    ])
    output1 = channel_as_first_dimension(img)
    output = channel_as_last_dimension(output1)

    assert (img == output).all()


def test_symmetry_start_channels_first_dimension():
    """
    Reshaping from convolution shape into RGB and back results in the same image
    """
    img = np.array([
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]
        ],
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3]
        ],
    ])
    output1 = channel_as_last_dimension(img)
    output = channel_as_first_dimension(output1)

    assert (img == output).all()
