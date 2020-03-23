import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from tadataka.camera import CameraParameters
from tadataka.vo.dvo.jacobian import (calc_projection_jacobian,
                                      calc_image_gradient,
                                      calc_jacobian)


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


def test_calc_jacobian():
    def run(j, gx, gy, p0, p1):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        assert_almost_equal(j[0], gx * (fx / z0))
        assert_almost_equal(j[1], gy * (fy / z0))
        assert_almost_equal(j[2],
                            (gx * (-fx * x0 / (z0 * z0)) +
                             gy * (-fy * y0 / (z0 * z0))))
        assert_almost_equal(j[3],
                            (gx * (-fx * x1 * y1 / (z1 * z1)) +
                             gy * (-fy * (1 + (y1 * y1) / (z1 * z1)))))
        assert_almost_equal(j[4],
                            (gx * (fx * (1 + (x1 * x1) / (z1 * z1))) +
                             gy * (fy * x1 * y1 / (z1 * z1))))
        assert_almost_equal(j[5],
                            gx * (-fx * y1 / z1) + gy * (fy * x1 / z1))

    P0 = np.array([[2, 4, -3],
                   [-4, 2, 3]])
    P1 = np.array([[3, -1, 2],
                   [2, -1, 5]])
    didx = np.array([10, -20])
    didy = np.array([-20, 10])

    fx, fy = 300, 400

    J = calc_jacobian([fx, fy], didx, didy, P0, P1)

    assert J.shape == (P0.shape[0], 6)

    for i in range(P0.shape[0]):
        run(J[i], didx[i], didy[i], P0[i], P1[i])
