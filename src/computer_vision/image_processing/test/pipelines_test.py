"""Tests for the .pipelines file"""
import numpy as np

from ..kernel import left
from ..pipelines import (
    padded_blur, padded_convolution_same_kernel,
    opening, closing, inner_border, outer_border
)


def test_blur_output_shape():
    """Padded blur creates an output of the same size as the input"""
    img = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [5, 6, 7, 8],
        [-1, -1, -1, -1]
    ])
    output = padded_blur(img)

    assert output.shape == img.shape


def test_blur_output_shape_3_chan():
    """Padded blur creates a 3D output of the same size as the input"""
    img = np.array([
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ],
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ],
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ]
    ])

    output = padded_blur(img)

    assert output.shape == img.shape


def test_padded_convolution_same_kernel_shape():
    """Padded convolution creates an output of the same size as the input"""
    img = np.array([
        [1, 2, 3, 4],
        [9, 9, 9, 9],
        [5, 6, 7, 8],
        [0, 0, 0, 0]
    ])
    output = padded_convolution_same_kernel(img, left)

    assert output.shape == img.shape


def test_padded_convolution_same_kernel_shape_3_chan():
    """Padded convolution of a 3D input creates an output of the same size"""
    img = np.array([
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ],
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ],
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
            [8, 7, 6, 5]
        ]
    ])
    output = padded_convolution_same_kernel(img, left)

    assert output.shape == img.shape


def test_opening():
    """Opening will disappear isolated pixels of higher value"""
    img = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1]  # <- see bottom right pixel
    ])
    middlestep, output = opening(img)

    assert middlestep.shape == img.shape
    assert output.shape == img.shape

    assert (middlestep == np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])).all()

    assert (output == np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0]  # <- bottom right pixel changed
    ])).all()


def test_closing():
    """Closing will eliminate lower value holes"""
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],  # <- see inner pixel
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    middlestep, output = closing(img)

    assert middlestep.shape == img.shape
    assert output.shape == img.shape

    assert (middlestep == np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])).all()

    assert (output == np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],  # <- inner pixel changed
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])).all()


def test_inner_border():
    """Inner border keeps the out-most pixels of the patterns in the input"""
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ])
    eroded, output = inner_border(img)

    assert eroded.shape == img.shape
    assert output.shape == img.shape

    assert (eroded == np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])).all()

    assert (output == np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ])).all()


def test_outer_border_shapes():
    """Outer border adds surrounding pixels to the patterns in the input"""
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    dilated, output = outer_border(img)

    assert dilated.shape == img.shape
    assert output.shape == img.shape

    assert (dilated == np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ])).all()

    assert (output == np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ])).all()
