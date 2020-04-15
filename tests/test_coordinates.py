import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tadataka.coordinates import (image_coordinates, xy_to_yx, yx_to_xy,
                                  get, substitute)


def test_image_coordinates():
    width, height = 3, 4
    coordinates = image_coordinates(image_shape=[height, width])
    assert_array_equal(
        coordinates,
#         x  y
        [[0, 0],
         [1, 0],
         [2, 0],
         [0, 1],
         [1, 1],
         [2, 1],
         [0, 2],
         [1, 2],
         [2, 2],
         [0, 3],
         [1, 3],
         [2, 3]]
    )


def test_xy_to_yx():
    coordinates = np.array([
        [0, 1],
        [2, 3],
        [4, 5]
    ])
    assert_array_equal(
        xy_to_yx(coordinates),
        [[1, 0],
         [3, 2],
         [5, 4]]
    )


def test_yx_to_xy():
    coordinates = np.array([
        [1, 0],
        [3, 2],
        [5, 4]
    ])
    assert_array_equal(
        yx_to_xy(coordinates),
        [[0, 1],
         [2, 3],
         [4, 5]]
    )


def test_substitute():
    width, height = 4, 5
    A = np.zeros((height, width))
    us = np.array([
    #    x  y
        [3, 4],
        [2, 1],
        [0, 0]
    ])
    values = np.array([1, 2, 3])
    A = substitute(A, us, values)
    assert_array_equal(
        A,
        [[3, 0, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]]
    )


def test_get():
    A = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ])
    us = np.array([
    #    x  y
        [1, 0],
        [0, 0],
        [4, 0],
        [1, 1],
        [2, 1],
    ])
    assert_array_equal(get(A, us), [1, 0, 4, 6, 7])
