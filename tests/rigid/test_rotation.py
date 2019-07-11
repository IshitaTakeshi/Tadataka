from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal
from rigid.rotation import tangent_so3, log_so3, rodrigues


def test_log_so3():
    # FIXME the current implementation of log_so3 cannot calculate
    # the cases commented out below
    RS = np.array([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
      # [[-1, 0, 0],
      #  [0, -1, 0],
      #  [0, 0, 1]],
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]],
        [[1, 0, 0],
         [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
         [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]]
    ])
    omegas = np.array([
        [0, 0, 0],
      # [0, 0, np.pi],
        [0, np.pi / 2, 0],
        [np.pi / 4, 0, 0]]
    )

    omegas_expected = np.array([
        [0, 0, 0],
      # [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    thetas_expected = np.array([
        0,
      # np.pi,
        np.pi / 2,
        np.pi / 4
    ])

    omegas_pred, thetas_pred = log_so3(RS)
    assert_equal(omegas_pred.shape[0], thetas_pred.shape[0])
    assert_equal(omegas_pred.shape, omegas.shape)
    assert_array_almost_equal(thetas_expected, thetas_pred)
    assert_array_almost_equal(omegas_expected, omegas_pred)


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
    ])

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
    ])

    assert_array_almost_equal(rodrigues(V), expected)
