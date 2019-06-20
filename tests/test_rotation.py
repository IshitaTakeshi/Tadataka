from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from rigid.rotation import tangent_so3, rodrigues


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
        [np.pi / 2, 0, 0],
        [0, -np.pi / 2, 0],
        [0, 0, np.pi],
        [-np.pi, 0, 0]
    ])

    expected = np.array([
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
    ])

    assert_array_almost_equal(rodrigues(V), expected)
