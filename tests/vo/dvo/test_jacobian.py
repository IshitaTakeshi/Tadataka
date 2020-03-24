import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from tadataka.camera import CameraParameters
from tadataka.vo.dvo.jacobian import calc_image_gradient, calc_jacobian


def test_calc_jacobian():
    def run(j, gx, gy, p1):
        x, y, z = p1
        assert_almost_equal(j[0], gx * (fx / z))
        assert_almost_equal(j[1], gy * (fy / z))
        assert_almost_equal(j[2],
                            (gx * (-fx * x / (z * z)) +
                             gy * (-fy * y / (z * z))))
        assert_almost_equal(j[3],
                            (gx * (-fx * x * y / (z * z)) +
                             gy * (-fy * (1 + (y * y) / (z * z)))))
        assert_almost_equal(j[4],
                            (gx * (fx * (1 + (x * x) / (z * z))) +
                             gy * (fy * x * y / (z * z))))
        assert_almost_equal(j[5],
                            gx * (-fx * y / z) + gy * (fy * x / z))

    P = np.array([[3, -1, 2],
                  [2, -1, 5]])
    didx = np.array([10, -20])
    didy = np.array([-20, 10])

    fx, fy = 300, 400

    J = calc_jacobian([fx, fy], didx, didy, P)

    assert J.shape == (P.shape[0], 6)

    for i in range(P.shape[0]):
        run(J[i], didx[i], didy[i], P[i])
