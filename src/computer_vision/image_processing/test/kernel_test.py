"""Tests the .kernel file"""
import numpy as np

from ..kernel import simple_gauss


def test_gauss_size():
    """simple_gauss creates a kernel of the right size"""
    kernel = simple_gauss(length=7)

    assert kernel.shape == (7, 7)


def test_gauss_size_default():
    """Default size of a gauss kernel is 5"""
    kernel = simple_gauss()

    assert kernel.shape == (5, 5)


def test_gauss_sigma():
    """Tests the right values of the default gauss kernel"""
    kernel = simple_gauss()

    expected = np.array([
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
        [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
        [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
    ])

    assert (np.round(kernel, 8) == expected).all()


def test_gauss_big_sigma():
    """Changing the gaussian sigma created the correct kernel"""
    kernel = simple_gauss(sigma=4.)

    expected = np.array([
        [0.03520395, 0.03866398, 0.0398913, 0.03866398, 0.03520395],
        [0.03866398, 0.04246407, 0.04381203, 0.04246407, 0.03866398],
        [0.0398913, 0.04381203, 0.04520277, 0.04381203, 0.0398913],
        [0.03866398, 0.04246407, 0.04381203, 0.04246407, 0.03866398],
        [0.03520395, 0.03866398, 0.0398913, 0.03866398, 0.03520395]
    ])

    assert (np.round(kernel, 8) == expected).all()
