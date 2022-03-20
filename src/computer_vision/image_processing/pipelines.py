"""
Pre-built pipelines that demonstrate image processing techniques.
Inputs must be an array of 2D images
- e.g. RGB channels must be the in the first dimension
"""
import numpy as np

from .kernel import simple_gauss
from .padding import padding
from .pooling import maxpool, minpool
from .convolution import convolution, conv_repeat_all_chan


def padded_blur(image: np.ndarray, length: int = 5, sigma: float = 1.):
    """Blurs an input using SAME padding"""
    kernel = simple_gauss(length, sigma)
    return padded_convolution_same_kernel(image, kernel)


def padded_convolution_same_kernel(image: np.ndarray, kernel: np.ndarray):
    """Uses the same kernel to convolute each channel of the input"""
    padding_size = (kernel.shape[0] - 1) // 2
    padded = padding(image, padding_size)

    if len(image.shape) == 3:
        output = conv_repeat_all_chan(padded, kernel)
    else:
        output = convolution(padded, kernel)
    return output


def opening(image: np.ndarray):
    """
    Opening: dilation of erosion.
    Returns both the erotion and its further dilation using SAME padding
    """
    assert len(image.shape) == 2

    eroded = minpool(padding(image, 1), 3)
    dilated = maxpool(padding(eroded, 1), 3)

    return eroded, dilated


def closing(image: np.ndarray):
    """
    Closing: erotion of dilation.
    Returns both the dilation and its further erotion using SAME padding
    """
    assert len(image.shape) == 2

    dilated = maxpool(padding(image, 1), 3)
    eroded: np.ndarray = minpool(padding(dilated, 1), 3)

    return dilated, eroded


def inner_border(image: np.ndarray):
    """
    Inner border: original - erotion.
    Returns both the erotion and the output using SAME padding
    """
    assert len(image.shape) == 2

    eroded = minpool(padding(image, 1), 3)
    border: np.ndarray = image - eroded
    return eroded, border


def outer_border(image: np.ndarray):
    """
    Outer border: dilation - image.
    Returns both the dilation and the output using SAME padding
    """
    assert len(image.shape) == 2

    dilated = maxpool(padding(image, 1), 3)
    border: np.ndarray = dilated - image
    return dilated, border
