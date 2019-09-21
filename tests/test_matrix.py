from autograd import numpy as np
from numpy.testing import (
    assert_array_almost_equal, assert_array_equal, assert_equal)

from vitamine.matrix import solve_linear, motion_matrix


def test_solve_linear():
    # some random matrix
    A = np.array(
        [[7, 3, 6, 7, 4, 3, 7, 2],
         [0, 1, 5, 2, 9, 5, 9, 7],
         [7, 5, 2, 3, 4, 1, 4, 3]]
    )
    x = solve_linear(A)
    assert_equal(x.shape, (8,))
    assert_array_almost_equal(np.dot(A, x), np.zeros(3))


def test_motion_matrix():
    R = np.arange(9).reshape(3, 3)
    t = np.arange(9, 12)
    T = motion_matrix(R, t)
    assert_array_equal(T,
        np.array([
            [0, 1, 2, 9],
            [3, 4, 5, 10],
            [6, 7, 8, 11],
            [0, 0, 0, 1]
        ])
    )
