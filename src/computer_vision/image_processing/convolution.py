"""Convolution functions and helpers"""
import numpy as np


def _multiply_sum(background: np.ndarray, kernel: np.ndarray) -> float:
    """
    Returns the sumation of multiplying each individual element
    of the background with the given kernel
    """
    assert background.shape == kernel.shape

    return (background * kernel).sum()


def conv_repeat_all_chan(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolves each channel of the input with the same kernel and
    returns their outputs as a separate channel each.
    """
    output = []
    for index in range(img.shape[0]):
        output.append(convolution_1d(img[index], kernel))
    return np.asarray(output)


def convolution_1d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Executes a 1-stride convolution with the given kernel, over a 2D input.
    The output shape will be (image_y - kernel_y + 1, image_x - kernel_x + 1)
    See other external packages for complete convolution functionality.
    """
    kernel_y, kernel_x = kernel.shape
    image_y, image_x = image.shape

    output_y = image_y - kernel_y + 1
    output_x = image_x - kernel_x + 1

    output = np.empty((output_y, output_x))
    assert output.shape == (output_y, output_x)

    for i in range(output_y):
        for j in range(output_x):
            output[i][j] = _multiply_sum(kernel,
                                         image[i:i + kernel_y, j:j + kernel_x])

    return output


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Executes a 3D or 2D convolution over input as needed.
    3D convolution requires a 3D filter, and creates a 2D output.
    """
    if len(image.shape) == 3:
        assert len(kernel.shape) == 3
        assert image.shape[0] == kernel.shape[0]

        channel_convs = []
        for index in range(image.shape[0]):
            channel_convs.append(convolution_1d(image[index], kernel[index]))

        output = np.asarray(channel_convs).sum(axis=0)
    else:
        output = convolution_1d(image, kernel)

    return output
