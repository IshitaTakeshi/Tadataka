import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation
from tadataka.visual_odometry.semi_dense import (
    convolve, coordinates_along_key_epipolar, coordinates_along_ref_epipolar
)


def test_convolve():
    def calc_error(a, b):
        return np.dot(a, b)

    A = np.array([5, 3, -4, 1, 0, 9, 6, -7])
    B = np.array([-1, 3, 1])
    errors = convolve(A, B, calc_error)
    assert_array_equal(errors, [0, -14, 7, 8, 33, 2])


def test_coordinates_along_key_epipolar():
    t_key_to_ref = np.array([20, 30, 10])  # [2, 3]
    x_key = np.array([4, 1])
    # d = [1, -1]
    disparity = np.sqrt(2)
    assert_array_almost_equal(
        coordinates_along_key_epipolar(x_key, t_key_to_ref, disparity),
        [[4-2, 1+2],
         [4-1, 1+1],
         [4+0, 1-0],
         [4+1, 1-1],
         [4+2, 1-2]]
    )


def test_coordinates_along_ref_epipolar():
    inv_depths = np.array([0.001, 0.01, 0.1])
    x_key = np.array([2.0, 1.2])
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    t = np.array([500, 200, 0])

    assert_array_almost_equal(
        coordinates_along_ref_epipolar(R, t, x_key, inv_depths),
        [[-0.75, -0.7],
         [-3.0, -1.6],
         [-25.5, -10.6]]
    )
