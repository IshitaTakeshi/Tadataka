import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.vo.semi_dense.epipolar import (
    reference_coordinates, key_coordinates_)


def test_reference_coordinates():
    width, height = 160, 200
    image_shape = [height, width]
    camera_model = CameraModel(
        CameraParameters(focal_length=[10, 10], offset=[80, 100]),
        distortion_model=None
    )

    search_step = 5.0
    x_min = np.array([-15.0, -20.0])
    x_max = np.array([15.0, 20.0])
    xs = reference_coordinates((x_min, x_max), search_step)

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
    t = np.array([-4.0, -8.0, 2.0])  # pi(t) == [-2, -4]
    pi_t = t[0:2] / t[2]
    x = np.array([7.0, 8.0])
    step_size = 5.0
    sampling_steps = np.array([-2, -1, 0, 1, 2])

    # step should be [3, 4]
    assert_array_almost_equal(
        key_coordinates_(x, pi_t, step_size, sampling_steps),
        [[7 - 2 * 3, 8 - 2 * 4],
         [7 - 1 * 3, 8 - 1 * 4],
         [7 - 0 * 3, 8 - 0 * 4],
         [7 + 1 * 3, 8 + 1 * 4],
         [7 + 2 * 3, 8 + 2 * 4]]
    )
