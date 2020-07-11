import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.matrix import motion_matrix
from tadataka.vo.semi_dense.epipolar import (
    key_epipolar_direction, key_coordinates,
    ref_coordinates, ref_search_range)
from tadataka.vo.semi_dense._epipolar import key_coordinates_
from tests.utils import random_rotation_matrix


def test_ref_coordinates():
    search_step = 5.0
    x_min = np.array([-15.0, -20.0])
    x_max = np.array([15.0, 20.0])
    xs = ref_coordinates((x_min, x_max), search_step)

    xs_true = np.array([
        [-15, -20],
        [-12, -16],
        [-9, -12],
        [-6, -8],
        [-3, -4],
        [0, 0],
        [3, 4],
        [6, 8],
        [9, 12],
        [12, 16]
    ])
    assert_array_equal(xs, xs_true)


def test_key_coordinates_():
    x_key = np.array([7., 8.])
    direction = np.array([9., 12.])
    step_size = 5.
    t_rk = np.array([-4., -8., 2.])

    expected = np.array([
        [7 - 2 * 3, 8 - 2 * 4],
        [7 - 1 * 3, 8 - 1 * 4],
        [7 - 0 * 3, 8 - 0 * 4],
        [7 + 1 * 3, 8 + 1 * 4],
        [7 + 2 * 3, 8 + 2 * 4]
    ], dtype=np.float64)
    # step should be [3, 4]
    assert_array_almost_equal(key_coordinates_(direction, x_key, step_size),
                              expected)
    assert_array_almost_equal(key_coordinates(t_rk, x_key, step_size),
                              expected)


def test_key_epipolar_direction():
    t_rk = np.array([3., 0., 6.])
    x_key = np.array([0., 0.5])
    assert_array_equal(key_epipolar_direction(t_rk, x_key), [-0.5, 0.5])


def test_ref_search_range():
    T_rk = motion_matrix(Rotation.from_rotvec([0., -np.pi/2, 0.]).as_matrix(),
                         np.array([2., 0., 2.]))
    x_key = np.array([0., 0.])
    search_range = 1.0, 3.0
    x_ref_min, x_ref_max = ref_search_range(T_rk, x_key, search_range)
    assert_array_almost_equal(x_ref_min, [0.5, 0.0])
    assert_array_almost_equal(x_ref_max, [-0.5, 0.0])
