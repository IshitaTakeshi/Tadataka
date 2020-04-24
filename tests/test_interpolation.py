import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.ndimage import map_coordinates
from skimage import data

import pytest

from tadataka.interpolation import _interpolation as cpp_interpolation
from tadataka.interpolation import interpolation


def test_interpolation():
    image = np.array([
        [0, 1, 5],
        [0, 0, 2],
        [4, 3, 2],
        [5, 6, 1]
    ], dtype=np.float64)
    # width, height = (3, 4)

    coordinates = np.array([[0.1, 1.2], [1.1, 2.1], [2.0, 2.3]])
    assert(interpolation(image, coordinates).shape == (3,))

    coordinate = np.array([0.1, 1.2])
    assert(interpolation(image, coordinate).dtype == np.float64)

    # ordinary
    expected = (image[2, 1] * (2.0 - 1.3) * (3.0 - 2.6) +
                image[2, 2] * (1.3 - 1.0) * (3.0 - 2.6) +
                image[3, 1] * (2.0 - 1.3) * (2.6 - 2.0) +
                image[3, 2] * (1.3 - 1.0) * (2.6 - 2.0))
    # 2d input
    coordinates = np.array([[1.3, 2.6]])
    assert_almost_equal(interpolation(image, coordinates).squeeze(), expected)
    # 1d input
    coordinate = np.array([1.3, 2.6])
    assert_almost_equal(interpolation(image, coordinate), expected)

    # minimum coordinate
    coordinate = np.array([0.0, 0.0])
    assert_almost_equal(interpolation(image, coordinate), image[0, 0])

    # minimum x
    coordinate = np.array([0.0, 0.1])
    expected = (image[0, 0] * (1.0 - 0.0) * (1.0 - 0.1) +
                image[1, 0] * (1.0 - 0.0) * (0.1 - 0.0))
    assert_almost_equal(interpolation(image, coordinate), expected)

    # minimum y
    coordinate = np.array([0.1, 0.0])
    expected = (image[0, 0] * (1.0 - 0.1) * (1.0 - 0.0) +
                image[0, 1] * (0.1 - 0.0) * (1.0 - 0.0))
    assert_almost_equal(interpolation(image, coordinate), expected)

    # maximum x
    coordinate = np.array([2.0, 2.9])
    expected = (image[2, 2] * (3.0 - 2.0) * (3.0 - 2.9) +
                image[3, 2] * (3.0 - 2.0) * (2.9 - 2.0))
    assert_almost_equal(interpolation(image, coordinate), expected)

    coordinate = np.array([1.9, 3.0])  # maximum y
    expected = (image[3, 1] * (2.0 - 1.9) * (4.0 - 3.0) +
                image[3, 2] * (1.9 - 1.0) * (4.0 - 3.0))
    assert_almost_equal(interpolation(image, coordinate), expected)

    # maximum coordinate
    coordinate = np.array([2.0, 3.0])
    assert_almost_equal(interpolation(image, coordinate), image[3, 2])

    with pytest.raises(ValueError):
        interpolation(image, [3.0, 2.01])

    with pytest.raises(ValueError):
        interpolation(image, [3.01, 2.0])

    with pytest.raises(ValueError):
        interpolation(image, [-0.01, 0.0])

    with pytest.raises(ValueError):
        interpolation(image, [0.0, -0.01])


def test_cpp_interpolation_():
    np.random.seed(3939)

    image = data.clock().astype(np.float64)
    height, width = image.shape

    N = 2000
    X = np.random.uniform(0, width-1, N)
    Y = np.random.uniform(0, height-1, N)
    C = np.column_stack((X, Y))

    expected = map_coordinates(image, np.vstack((Y, X)), order=1)
    intensities = cpp_interpolation.interpolation(image, C)
    assert_array_almost_equal(intensities, expected)

    C = np.array([[width-1, 0],
                  [0, height-1],
                  [width-1, height-1]])
    intensities = cpp_interpolation.interpolation(image, C.astype(np.float64))
    expected = image[C[:, 1], C[:, 0]]
    assert_array_almost_equal(intensities, expected)
