import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal

from tadataka.so3 import is_rotation_matrix, exp_so3, log_so3, tangent_so3


def test_is_rotation_matrix():
    M = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    assert(is_rotation_matrix(M))

    M = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-1, 0, 0]])
    assert(is_rotation_matrix(M))

    M = np.array([[1, 0, 0],
                  [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                  [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]])
    assert(is_rotation_matrix(M))

    M = np.array([[-7 / 25, 0, -24 / 25],
                  [0, -1, 0],
                  [-24 / 25, 0, 7 / 25]])
    assert(is_rotation_matrix(M))

    M = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-2, 0, 0]])
    assert(not is_rotation_matrix(M))

    M = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-1, 1, 0]])
    assert(not is_rotation_matrix(M))


def test_tangent_so3():
    assert_array_equal(tangent_so3([1, 2, 3]),
                       [[0, -3, 2],
                        [3, 0, -1],
                        [-2, 1, 0]])

    assert_array_equal(tangent_so3([4, 5, 6]),
                       [[0, -6, 5],
                        [6, 0, -4],
                        [-5, 4, 0]])
