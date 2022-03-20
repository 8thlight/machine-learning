"""Maxpool and Minpool functionality"""
import numpy as np


def pool(image: np.ndarray, size: int, method) -> np.ndarray:
    """Pools the given area"""
    image_y, image_x = image.shape
    output_y = image_y - size + 1
    output_x = image_x - size + 1
    output = np.empty((output_y, output_x), int)

    for i in range(output_y):
        for j in range(output_x):
            output[i][j] = method(image[i:i + size, j:j + size])

    return output


def maxpool(image: np.ndarray, size: int) -> np.ndarray:
    """Returns maximum values in the areas of size NxN"""
    return pool(image, size, np.amax)


def minpool(image: np.ndarray, size: int) -> np.ndarray:
    """Returns minimum values in the areas of size NxN"""
    return pool(image, size, np.amin)
