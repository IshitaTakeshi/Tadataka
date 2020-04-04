import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from tadataka.math import weighted_mean, solve_linear_equation


def test_weighted_mean():
    x = np.array([2.0, 3.0])
    w = np.array([0.9, 0.1])
    assert_almost_equal(weighted_mean(x, w), 1.8 + 0.3)

    x = np.array([2.0, 3.0])
    w = np.array([0.4, 0.8])
    assert_almost_equal(weighted_mean(x, w), (0.8 + 2.4) / 1.2)


def test_solve_linear_equation():
    A = np.array([
        [0, 1],
        [1, 1],
        [2, 1],
    ], dtype=np.float64)
    b = np.array([6, 0, 0], dtype=np.float64)

    x = solve_linear_equation(A, b, weights=None)
    assert_array_almost_equal(x, [-3, 5])

    A = np.random.uniform(-1, 1, (10, 4))
    weights = np.random.random(10)
    b = np.random.uniform(-1, 1, 10)
    x = solve_linear_equation(A, b, weights)
    W = np.diag(weights)
    assert_array_almost_equal((A.T @ W @ A) @ x, A.T @ W @ b)
