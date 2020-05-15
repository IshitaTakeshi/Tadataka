from numpy.testing import assert_array_almost_equal
import numpy as np

from tadataka.transform_project import (
    exp_so3, transform_project, pose_jacobian, point_jacobian
)

np.random.seed(3939)


def test_transform_project():
    points = np.array([
        [4., -2., -3.],
        [-3., 2., 1.],
        [5., 3., -6.]
    ])

    omegas = np.array([
        [0., 0., 0.],         # R : (x, y, z) -> (x, y, z)
        [np.pi / 2., 0., 0.],  # R : (x, y, z) -> (x, -z, y)
        [0., 0., np.pi]       # R : (x, y, z) -> (-x, -y, z)
    ])

    translations = np.array([
        [4., -8., 1.],
        [3., -1., 2.],
        [2., 0., -4.]
    ])

    expected = np.array([
        [-4.0, 5.0],
        [0.0, -0.5],
        [0.3, 0.3]
    ])

    for i, (omega, t, point) in enumerate(zip(omegas, translations, points)):
        pose = np.concatenate((omega, t))

        assert_array_almost_equal(
            transform_project(pose, point),
            expected[i]
        )


def test_exp_so3():
    V = np.array([
        [0., 0., 0.],
        [np.pi / 2, 0., 0.],
        [0., -np.pi / 2., 0.],
        [0., 0., np.pi],
        [-np.pi, 0., 0.]
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

    for v, R_true in zip(V, expected):
        assert_array_almost_equal(exp_so3(v), R_true)


def sum_squared(v):
    return (v * v).sum()


def test_point_jacobian():
    pose = np.random.random(6)

    point = np.random.random(3)
    dpoint = 0.01 * np.random.random(3)

    f0 = transform_project(pose, point)
    f1 = transform_project(pose, point + dpoint)
    df = f1 - f0

    J = point_jacobian(pose, point)

    assert(sum_squared(df - J.dot(dpoint)) < 1e-3 * sum_squared(df))


def test_point_jacobian():
    pose = np.random.random(6)
    dpose = 0.01 * np.random.random(6)

    point = np.random.random(3)

    f0 = transform_project(pose, point)
    f1 = transform_project(pose + dpose, point)
    df = f1 - f0

    J = pose_jacobian(pose, point)

    assert(sum_squared(df - J.dot(dpose)) < 1e-3 * sum_squared(df))
