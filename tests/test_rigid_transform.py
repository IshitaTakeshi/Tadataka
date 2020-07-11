import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from scipy.spatial.transform import Rotation

from tadataka.matrix import motion_matrix
from tadataka.rigid_transform import (inv_transform_all, transform_all,
                                      transform_each, Transform, transform_se3)


def test_transform_each():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [2, -3, 5],   # [ 1, -5,  2] + [ 1,  2,  3]
        [1, 3, 10]    # [ -3, -2, 4] + [ 4,  5,  6]
    ])

    assert_array_equal(
        transform_each(rotations, translations, points),
        expected
    )


def test_transform_all():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
        [0, 0, 6]
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [[2, -3, 5],   # [ 1, -5,  2] + [ 1,  2,  3]
         [5, -1, 1],   # [ 4, -3, -2] + [ 1,  2,  3]
         [1, -4, 3]],  # [ 0, -6,  0] + [ 1,  2,  3]
        [[-1, 7, 7],   # [-5,  2,  1] + [ 4,  5,  6]
         [1, 3, 10],   # [-3, -2,  4] + [ 4,  5,  6]
         [-2, 5, 6]]   # [-6,  0,  0] + [ 4,  5,  6]
    ])

    assert_array_equal(transform_all(rotations, translations, points),
                       expected)


def test_inv_transform_all():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
        [0, 0, 6]
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    # [R.T for R in rotations]
    # [[1, 0, 0],
    #  [0, 0, 1],
    #  [0, -1, 0]],
    # [[0, 0, 1],
    #  [0, 1, 0],
    #  [-1, 0, 0]]

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # p - t
    # [[0, 0, 2],
    #  [3, -4, 0],
    #  [-1, -2, 3]],
    # [[-3, -3, -1],
    #  [0, -7, -3],
    #  [-4, -5, 0]]

    # np.dot(R.T, p-t)
    expected = np.array([
        [[0, 2, 0],
         [3, 0, 4],
         [-1, 3, 2]],
        [[-1, -3, 3],
         [-3, -7, 0],
         [0, -5, 4]]
    ])

    assert_array_equal(inv_transform_all(rotations, translations, points),
                       expected)


def test_transform_class():
    P = np.array([
        [1, 2, 5],
        [4, -2, 3],
    ])

    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    t = np.array([1, 2, 3])

    assert_array_equal(
        Transform(R, t, s=1.0)(P),
        [[2, -3, 5],    # [   1  -5   2] + [   1   2   3]
         [5, -1, 1]]    # [   4  -3  -2] + [   1   2   3]
    )

    assert_array_equal(
        Transform(R, t, s=0.1)(P),
        [[1.1, 1.5, 3.2],    # [   0.1  -0.5   0.2] + [   1   2   3]
         [1.4, 1.7, 2.8]]    # [   0.4  -0.3  -0.2] + [   1   2   3]
    )


def test_transform_se3():
    R_10 = np.random.random((3, 3))
    t_10 = np.random.random(3)
    T_10 = motion_matrix(R_10, t_10)

    P0 = np.random.uniform(-10, 10, (10, 3))
    P1 = transform_se3(T_10, P0)
    assert_array_almost_equal(P1, np.dot(R_10, P0.T).T + t_10)
