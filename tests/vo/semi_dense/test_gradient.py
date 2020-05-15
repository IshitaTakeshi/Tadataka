import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from tadataka.vo.semi_dense._gradient import calc_gradient_norm, gradient1d
from tadataka.vo.semi_dense.gradient import GradientImage


def test_gradient_image():
    width, height = 6, 4
    grad_x = np.random.uniform(-10, 10, (height, width))
    grad_y = np.random.uniform(-10, 10, (height, width))
    gradient_image = GradientImage(grad_x, grad_y)

    u_key = np.array([4.3, 2.1])
    gx, gy = gradient_image(u_key)


    u, v = 4, 2

    expected_x = (0.7 * 0.9 * grad_x[v, u] +
                  0.3 * 0.9 * grad_x[v, u+1] +
                  0.7 * 0.1 * grad_x[v+1, u] +
                  0.3 * 0.1 * grad_x[v+1, u+1])
    assert_almost_equal(gx, expected_x)

    expected_y = (0.7 * 0.9 * grad_y[v, u] +
                  0.3 * 0.9 * grad_y[v, u+1] +
                  0.7 * 0.1 * grad_y[v+1, u] +
                  0.3 * 0.1 * grad_y[v+1, u+1])
    assert_almost_equal(gy, expected_y)


def test_gradient1d():
    intensities = np.array([-1, 1, 0, 3, -2])

    assert_array_equal(gradient1d(intensities),
                       [1 - (-1), 0 - 1, 3 - 0, -2 - 3])


def test_calc_gradient():
    intensities = np.array([-1, 1, 0, 3, -2])
    assert_almost_equal(calc_gradient_norm(intensities),
                        np.linalg.norm([1 - (-1), 0 - 1, 3 - 0, -2 - 3]))
