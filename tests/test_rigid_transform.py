import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from scipy.spatial.transform import Rotation
from tadataka.pose import WorldPose
from tadataka.rigid_transform import (inv_transform_all, transform_all,
                                      transform_each, Transform, Warp3D)


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


def test_transform():
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


def test_warp3d():
    # rotate (3 / 4) * pi around the y-axis
    rotation0 = Rotation.from_rotvec([0, (3 / 4) * np.pi, 0])
    t0 = np.array([0, 0, 3])
    pose0 = WorldPose(rotation0, t0)

    # rotate - pi / 2 around the y-axis
    rotation1 = Rotation.from_rotvec([0, -np.pi / 2, 0])
    t1 = np.array([4, 0, 3])
    pose1 = WorldPose(rotation1, t1)

    warp01 = Warp3D(pose0, pose1)

    P0 = np.array([
        [0, 0, 2 * np.sqrt(2)],
        [0, 0, 4 * np.sqrt(2)]
    ])

    # world point should be at
    # [[2, 0, 1], [4, 0, 7]]
    # [[2, 0, -2], [0, 0, 4]]
    assert_array_almost_equal(warp01(P0), [[-2, 0, 2], [-4, 0, 0]])

    # accept 1d input
    # center of camera 0 seen from camera 1
    assert_array_almost_equal(warp01(np.zeros(3)), [0, 0, 4])
