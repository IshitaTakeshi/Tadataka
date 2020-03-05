from numpy.testing import assert_almost_equal
import numpy as np
from numpy.linalg import norm

from tadataka.vo.semi_dense.variance import (
    alpha_index, calc_alpha_,
    geometric_variance, photometric_variance
)


def test_geomteric_disparity_error():
    t = np.array([-8, 6, 2])
    pi_t = t[0:2] / t[2]
    sigma_l = 0.5

    gradient = np.array([2, -3])
    x = np.array([2, 5])

    variance = geometric_variance(x, pi_t, gradient, sigma_l)

    direction = np.array([6, 2])  # x - pi(t)
    d = np.dot(direction / norm(direction), gradient / norm(gradient))
    assert_almost_equal(variance, (sigma_l * sigma_l) / (d * d))


def test_photometric_variance():
    sigma_i = 2.0
    gradient = 10.0
    variance = photometric_variance(gradient, sigma_i)
    assert(variance == 0.8)


def test_calc_alpha():
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
