"""Common kernels used in convolutions"""
import numpy as np

top = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

bottom = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

left = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

right = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

top_sobel = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

bottom_sobel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

left_sobel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

right_sobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

outline = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])


def simple_gauss(length=5, sigma=1.) -> np.ndarray:
    """
    Creates a gaussian filter of the desired NxN size and the standard deviation
    It is created by calculating the outer product of two gaussian vectors.
    """
    integer_space = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    gauss = np.exp(-0.5 * np.square(integer_space) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def from_name(name: str):
    """Returns the desired kernel given its name"""
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "top_sobel": top_sobel,
        "bottom_sobel": bottom_sobel,
        "left_sobel": left_sobel,
        "right_sobel": right_sobel,
        "sharpen": sharpen,
        "outline": outline
    }[name]
