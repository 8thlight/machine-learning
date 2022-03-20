"""
RGB and BGR images have the colors in the last dimension of the array.
Convolutions need each channel to be in the first dimension, so that
accessing an element with `img[0]` returns a complete 2D version of
the channel 0.

This file provides functionality to reshape images to the required shape
and back.
"""
import numpy as np


def channel_as_first_dimension(img: np.ndarray) -> np.ndarray:
    """
    Moves channel from the last dimension to the first.
    - e.g. RGB to convolution shape
    """
    assert len(img.shape) == 3

    temp = img.flatten()
    new_shape = (img.shape[2], img.shape[0], img.shape[1])

    return temp.reshape(new_shape, order="F")


def channel_as_last_dimension(img: np.ndarray) -> np.ndarray:
    """
    Moves channel from the first dimension to the last.
    - e.g. convolution shape to RGB
    """
    assert len(img.shape) == 3

    temp = img.flatten()
    new_shape = (img.shape[1], img.shape[2], img.shape[0])

    return temp.reshape(new_shape, order="F")
