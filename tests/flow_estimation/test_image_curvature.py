import numpy as np
from numpy.testing import assert_array_equal

from tadataka.flow_estimation.image_curvature import (
    grad_x, grad_y, compute_curvature, compute_image_curvature)


A = np.arange(25).reshape(5, 5)
B = np.arange(49).reshape(7, 7)


def test_grad_x():
    GX = grad_x(A)
    assert_array_equal(GX[1:4, 1:4], 8)


def test_grad_y():
    GY = grad_y(A)
    assert_array_equal(GY[1:4, 1:4], 40)


def test_compute_image_curvature():
    G = compute_image_curvature(B)
    assert_array_equal(G[2:5, 2:5], 0)


def test_compute_curvature():
    fx = np.array([2, 1])
    fy = np.array([1, -2])
    fxx = np.array([-3, 1])
    fyx = np.array([2, 4])
    fxy = np.array([4, 3])
    fyy = np.array([-2, 4])

    expected = np.array([
        1 * (-3) - 2 * 1 * 4 - 1 * 2 * 2 + 4 * (-2),
        4 * 1 - 1 * (-2) * 3 - (-2) * 1 * 4 + 1 * 4
    ])

    assert_array_equal(
        compute_curvature(fx, fy, fxx, fxy, fyx, fyy),
        expected
    )
