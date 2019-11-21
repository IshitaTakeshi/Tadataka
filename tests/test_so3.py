import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal

from tadataka.so3 import (is_rotation_matrix, tangent_so3,
                          inv_rodrigues, rodrigues)

from tadataka.so3_codegen import exp_so3


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


def test_inv_rodrigues():
    RS = np.array([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]],
        [[-1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]],
        [[-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]],
        [[1, 0, 0],
         [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
         [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]],
        [[1 / 2, 1 / np.sqrt(2), 1 / 2],
         [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
         [1 / 2, -1 / np.sqrt(2), 1 / 2]],
        [[-1, 0, 0],
         [0, 0, -1],
         [0, -1, 0]],
        [[-7 / 25, 0, -24 / 25],
         [0, -1, 0],
         [-24 / 25, 0, 7 / 25]]
    ])
    omegas = np.array([
        [0, 0, 0],
        [np.pi, 0, 0],
        [0, np.pi, 0],
        [0, 0, np.pi],
        [0, np.pi / 2, 0],
        [np.pi / 4, 0, 0],
        [-np.pi / np.sqrt(8), 0, -np.pi / np.sqrt(8)],
        [0, np.pi / np.sqrt(2), -np.pi / np.sqrt(2)],
        [3 * np.pi / 5, 0, -4 * np.pi / 5],
    ])
    assert_array_almost_equal(rodrigues(omegas), RS)
    assert_array_almost_equal(omegas, inv_rodrigues(RS))


def test_tangents_so3():
    V = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [[0, -3, 2],
         [3, 0, -1],
         [-2, 1, 0]],
        [[0, -6, 5],
         [6, 0, -4],
         [-5, 4, 0]]
    ])

    W = tangent_so3(V)
    assert_array_equal(W, expected)


def test_rodrigues():
    V = np.array([
        [0, 0, 0],
        [np.pi / 2, 0, 0],
        [0, -np.pi / 2, 0],
        [0, 0, np.pi],
        [-np.pi, 0, 0]
    ], dtype=np.float64)

    expected = np.array([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]],
        [[-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]]
    ], dtype=np.float64)

    assert_array_almost_equal(rodrigues(V), expected)

    for v, R_true in zip(V, expected):
        assert_array_almost_equal(exp_so3(v), R_true)
