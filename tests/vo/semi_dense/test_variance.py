from numpy.testing import assert_almost_equal
import numpy as np
from numpy.linalg import norm

from tadataka.projection import pi
from tadataka.matrix import motion_matrix, to_homogeneous, from_homogeneous
from tadataka.vo.semi_dense.variance import (
    alpha_index, calc_alpha_, calc_alpha,
    geometric_variance, photometric_variance
)


def test_geomteric_variance():
    sigma_l = 0.5
    epsilon = 1e-4

    gradient = np.array([20, -30])
    direction = np.array([6, 2])  # epipolar line direction

    # smoke
    variance = geometric_variance(direction, gradient, sigma_l, epsilon)
    p = np.dot(direction / norm(direction), gradient / norm(gradient))
    assert_almost_equal(variance, (sigma_l * sigma_l) / (p * p))

    # zero epipolar direction
    variance = geometric_variance(np.zeros(2), gradient, sigma_l, epsilon)
    assert_almost_equal(variance, (sigma_l * sigma_l) / epsilon)

    # zero gradient
    variance = geometric_variance(direction, np.zeros(2), sigma_l, epsilon)
    assert_almost_equal(variance, (sigma_l * sigma_l) / epsilon)

    # the case gradient is orthogonal to epipolar direction (max variance)
    gradient = np.array([direction[1], -direction[0]])
    variance = geometric_variance(direction, gradient, sigma_l, epsilon)
    assert_almost_equal(variance, (sigma_l * sigma_l) / epsilon)


def test_photometric_variance():
    sigma_i = 2.0
    gradient = 10.0
    variance = photometric_variance(gradient, sigma_i)
    assert(variance == 0.8)


def test_calc_alpha_():
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    t = np.array([2, 4, -3])

    direction = np.array([0.1, 0.3])
    x_key = np.array([0.3, 0.9])
    x_ref = np.array([-0.6, 0.4])

    X = np.append(x_key, 1)

    alpha = calc_alpha_(x_key, x_ref[0], direction[0], R[0], R[2], t[0], t[2])

    n = t[0] * np.dot(R[2], X) - t[2] * np.dot(R[0], X)
    d = t[0] - x_ref[0] * t[2]
    assert(alpha == direction[0] * n / (d * d))

    alpha = calc_alpha_(x_key, x_ref[1], direction[1], R[1], R[2], t[1], t[2])

    n = t[1] * np.dot(R[2], X) - t[2] * np.dot(R[1], X)
    d = t[1] - x_ref[1] * t[2]
    assert(alpha == direction[1] * n / (d * d))


def test_alpha_index():
    assert(alpha_index([1.9, 0.2]) == 0)
    assert(alpha_index([0.9, -1.2]) == 1)
    assert(alpha_index([-0.9, -0.2]) == 0)


def test_calc_alpha():
    R_rk = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    t_rk = np.array([2, 4, -3])

    T_rk = motion_matrix(R_rk, t_rk)

    x_key = np.array([0.3, 0.9])
    prior_inv_depth = 0.1

    P_key = (1 / prior_inv_depth) * to_homogeneous(x_key)
    x_ref = pi(from_homogeneous(np.dot(T_rk, to_homogeneous(P_key))))

    direction = np.array([0.1, 0.3])
    assert_almost_equal(
        calc_alpha(T_rk, x_key, direction, prior_inv_depth),
        calc_alpha_(x_key, x_ref[1], direction[1],
                    R_rk[1], R_rk[2], t_rk[1], t_rk[2]))

    direction = np.array([-2.0, 1.0])
    assert_almost_equal(
        calc_alpha(T_rk, x_key, direction, prior_inv_depth),
        calc_alpha_(x_key, x_ref[0], direction[0],
                    R_rk[0], R_rk[2], t_rk[0], t_rk[2]))
