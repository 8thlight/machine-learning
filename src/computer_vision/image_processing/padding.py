"""Padding functionality"""
import numpy as np


def padding_1d(image: np.ndarray, length: int) -> np.ndarray:
    """Adds 0 padding to a 2D input"""
    img_y, img_x = image.shape
    output = np.zeros((img_y + 2 * length, img_x + 2 * length), int)

    for i in range(img_y):
        for j in range(img_x):
            output[i + length][j + length] = image[i][j]

    return output


def padding(image: np.ndarray, size: int) -> np.ndarray:
    """Adds 0 padding to a 3D or 2D input as needed"""
    if len(image.shape) == 3:
        output = []
        for index in range(image.shape[0]):
            output.append(padding_1d(image[index], size))

        output = np.asarray(output)
    else:
        output = padding_1d(image, size)

    return output
