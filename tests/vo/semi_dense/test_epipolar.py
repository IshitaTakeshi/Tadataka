import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.vo.semi_dense.epipolar import (
    ReferenceCoordinates, NormalizedKeyCoordinates, EpipolarDirection)


def test_reference_coordinates():
    width, height = 160, 200
    image_shape = [height, width]
    camera_model = CameraModel(
        CameraParameters(focal_length=[10, 10], offset=[80, 100]),
        FOV(0.00)
    )

    search_step = 5.0
    coordinates = ReferenceCoordinates(camera_model, image_shape, search_step)

    x_min = np.array([-15.0, -20.0])
    x_max = np.array([15.0, 20.0])
    xs, us = coordinates((x_min, x_max))

    mask = [3, 4, 5, 6, 7]  # only masked coordinates are in the image

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
    assert_array_equal(xs, xs_true[mask])

    us_true = np.array([
        [-70, -100],
        [-40, -60],
        [-10, -20],
        [20, 20],
        [50, 60],
        [80, 100],
        [110, 140],
        [140, 180],
        [170, 220],
        [200, 260]
    ])
    assert_array_equal(us, us_true[mask])


def test_normalized_key_intensities():
    t = np.array([-4, -8, 2])  # x0 = [-2, -4]
    x = [7, 8]
    search_step = 5
    # step should be [3, 4]
    sampling_steps = [-2, -1, 0, 1, 2]

    direction = EpipolarDirection(t)
    coordinates = NormalizedKeyCoordinates(direction, sampling_steps)

    assert_array_almost_equal(
        coordinates(x, search_step),
        [[7 - 2 * 3, 8 - 2 * 4],
         [7 - 1 * 3, 8 - 1 * 4],
         [7 - 0 * 3, 8 - 0 * 4],
         [7 + 1 * 3, 8 + 1 * 4],
         [7 + 2 * 3, 8 + 2 * 4]]
    )
