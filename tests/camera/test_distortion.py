from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.spatial.transform import Rotation

from tadataka.camera.distortion import (
    FOV, fov_distort_factors, fov_undistort_factors,
    NoDistortion, RadTan)
from tadataka.camera.model import CameraModel
from tadataka.camera.normalizer import Normalizer
from tadataka.camera.parameters import CameraParameters
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform
from tadataka.pose import solve_pnp
from tadataka.so3 import exp_so3
from tests.data import dummy_points as points


X = np.array([
    [0.0, 0.0],
    [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],  # r = 1
    [1.0, np.sqrt(3)]  # r = 2
])


def test_no_distortion():
    assert_array_equal(X, NoDistortion().distort(X))
    assert_array_equal(X, NoDistortion().undistort(X))


def test_fov_undistort():
    # tan(pi / 6) = 1 / sqrt(3) = 3 / sqrt(3)
    expected = np.array([
        # (pi / 3) / (2 * tan(pi / 6))
        (np.pi / 3) / (2 * np.sqrt(3) / 3),
        # tan(1 * pi / 3) / (2 * 1 * tan(pi / 6))
        np.sqrt(3) / (2 * np.sqrt(3) / 3),
        # tan(2 * pi / 3) / (2 * 2 * tan(pi / 6))
        -np.sqrt(3) / (2 * 2 * np.sqrt(3) / 3)
    ])

    # test 'calc_fov_undistort_factors' separately because
    # we cannot test factors for X[0] using FOV.undistort
    assert_array_almost_equal(fov_undistort_factors(X, omega=np.pi/3), expected)
    assert_array_almost_equal(
        FOV(omega=np.pi/3).undistort(X),
        expected.reshape(-1, 1) * X
    )

    # all factors = 1
    assert_array_almost_equal(FOV(omega=0).undistort(X), X)


def test_fov_distort():
    expected = np.array([
        # (2 * tan(pi / 6)) / (pi / 3)
        (2 * np.sqrt(3) / 3) / (np.pi / 3),
        # arctan(2 * r * tan(pi / 6)) / (pi / 3)
        np.arctan(2 * np.sqrt(3) / 3) / (np.pi / 3),
        # arctan(2 * 2 * tan(pi / 6)) / (pi / 3)
        np.arctan(2 * 2 * np.sqrt(3) / 3) / (np.pi / 3)
    ])

    assert_array_almost_equal(fov_distort_factors(X, omega=np.pi/3), expected)
    assert_array_almost_equal(
        FOV(omega=np.pi/3).distort(X),
        expected.reshape(-1, 1) * X
    )

    assert_array_almost_equal(FOV(omega=0).distort(X), X)


def test_radtan_distort():
    X = np.array([
        [0, 1],
        [0, 2],
        [1, 0],
        [2, 0],
        [1, 2]
    ])
    r2 = np.array([1, 4, 1, 4, 5]).reshape(-1, 1)
    r4 = np.array([1, 16, 1, 16, 25]).reshape(-1, 1)
    r6 = np.array([1, 64, 1, 64, 125]).reshape(-1, 1)

    assert_array_equal(RadTan([2, 0, 0, 0]).distort(X),
                       X * (1 + 2 * r2))

    assert_array_equal(RadTan([0, 3, 0, 0]).distort(X),
                       X * (1 + 3 * r4))

    assert_array_equal(RadTan([0, 0, 0, 0, 2]).distort(X),
                       X * (1 + 2 * r6))

    Y = RadTan([0, 0, 3, 0]).distort(X)
    # X[:, 0] + 2 * p1 * X[:, 0] * X[:, 1]
    assert_array_equal(Y[:, 0],
                       [0 + 0,
                        0 + 0,
                        1 + 0,
                        2 + 0,
                        1 + 2 * 3 * 1 * 2])
    # X[:, 1] + p1 * (r2 + 2 * X[:, 1] * X[:, 1])
    assert_array_equal(Y[:, 1],
                       [1 + 3 * (1 + 2 * 1 * 1),
                        2 + 3 * (4 + 2 * 2 * 2),
                        0 + 3 * (1 + 2 * 0 * 0),
                        0 + 3 * (4 + 2 * 0 * 0),
                        2 + 3 * (5 + 2 * 2 * 2)])

    Y = RadTan([0, 0, 0, 4]).distort(X)
    # X[:, 0] + p2 * (r2 + 2 * X[:, 0] * X[:, 0])
    assert_array_equal(Y[:, 0],
                       [0 + 4 * (1 + 2 * 0 * 0),
                        0 + 4 * (4 + 2 * 0 * 0),
                        1 + 4 * (1 + 2 * 1 * 1),
                        2 + 4 * (4 + 2 * 2 * 2),
                        1 + 4 * (5 + 2 * 1 * 1)])
    # X[:, 1] + 2 * p2 * X[:, 0] * X[:, 1]
    assert_array_equal(Y[:, 1],
                       [1 + 0,
                        2 + 0,
                        0 + 0,
                        0 + 0,
                        2 + 2 * 4 * 1 * 2])

    assert_array_equal(RadTan([0, 0, 0, 0]).distort(X), X)


def test_undistort():
    X_true = np.array([
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [2, 0],
        [1, 2],
        [-3, -5],
        [9, -3]
    ])

    model = RadTan([0.5, 0.2, -0.3, 0.1, 0.02])
    Y = model.distort(X_true)
    X_pred = model.undistort(Y)

    assert_array_almost_equal(X_true, X_pred, decimal=4)


def test_eq():
    assert(FOV(0.1) == FOV(0.1))
    assert(FOV(0.1) != FOV(0.2))
    assert(RadTan([0, 0, 0, 0]) != FOV(0))
