import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.matrix import motion_matrix
from tadataka.vo.semi_dense.epipolar import (
    key_coordinates_, key_epipolar_direction, key_coordinates,
    ref_coordinates, ref_search_range)
from tests.utils import random_rotation_matrix


def test_ref_coordinates():
    width, height = 160, 200
    image_shape = [height, width]
    camera_model = CameraModel(
        CameraParameters(focal_length=[10, 10], offset=[80, 100]),
        distortion_model=None
    )

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


def test_key_coordinates():
    x = np.array([7.0, 8.0])
    direction = np.array([9, 12])
    step_size = 5.0

    # step should be [3, 4]
    assert_array_almost_equal(
        key_coordinates(direction, x, step_size),
        [[7 - 2 * 3, 8 - 2 * 4],
         [7 - 1 * 3, 8 - 1 * 4],
         [7 - 0 * 3, 8 - 0 * 4],
         [7 + 1 * 3, 8 + 1 * 4],
         [7 + 2 * 3, 8 + 2 * 4]]
    )


def test_key_epipolar_direction():
    t_rk = np.array([3, 0, 6])
    x_key = np.array([0, 0.5])
    assert_array_equal(key_epipolar_direction(t_rk, x_key), [-0.5, 0.5])
