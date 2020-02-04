import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


from tadataka.camera import CameraParameters
from tadataka.jacobian import calc_projection_jacobian, calc_image_gradient


def test_calc_projection_jacobian():
    camera_parameters = CameraParameters(
        focal_length=[3, 2],
        offset=[0, 0]
    )

    P = np.array([
        [1, 6, 2],
        [8, 5, 3],
    ])

    J = calc_projection_jacobian(camera_parameters, P)

    GT = np.array([
        # x' = 1, y' = 6, z' = 2, fx = 3, fy = 2
        [[3 / 2, 0, -3 * 1 / 4, -3 * 1 * 6 / 4, 3 * (1 + 1 / 4), -3 * 6 / 2],
         [0, 2 / 2, -2 * 6 / 4, -2 * (1 + 36 / 4), 2 * 1 * 6 / 4, 2 * 1 / 2]],
        # x' = 8, y' = 5, z' = 3, fx = 3, fy = 2
        [[3 / 3, 0, -3 * 8 / 9, -3 * 8 * 5 / 9, 3 * (1 + 64 / 9), -3 * 5 / 3],
         [0, 2 / 3, -2 * 5 / 9, -2 * (1 + 25 / 9), 2 * 8 * 5 / 9, 2 * 8 / 3]]
    ])

    assert_array_equal(J, GT)
