import itertools

import numpy as np
from numpy.linalg import inv

from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal, assert_equal)

from scipy.spatial.transform import Rotation

from tadataka.camera import CameraParameters
from tadataka.matrix import (
    solve_linear, motion_matrix, inv_motion_matrix, get_rotation_translation,
    decompose_essential, estimate_fundamental, fundamental_to_essential,
    to_homogeneous, from_homogeneous, calc_relative_transform)
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform
from tadataka.so3 import tangent_so3

from tests.utils import random_rotation_matrix


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


def test_to_homogeneous():
    assert_array_equal(
        to_homogeneous(np.array([[2, 3], [4, 5]], dtype=np.float64)),
        [[2, 3, 1], [4, 5, 1]]
    )

    assert_array_equal(
        to_homogeneous(np.array([2, 3], dtype=np.float64)),
        [2, 3, 1]
    )


def test_from_homogeneous():
    assert_array_equal(from_homogeneous(np.array([2, 3, 1])), [2, 3])
    assert_array_equal(from_homogeneous(np.array([[2, 3, 1], [3, 4, 1]])),
                       [[2, 3], [3, 4]])


def test_motion_matrix():
    R = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    t = np.array([9, 10, 11])
    T = motion_matrix(R, t)
    assert_array_equal(T,
        np.array([
            [0, 1, 2, 9],
            [3, 4, 5, 10],
            [6, 7, 8, 11],
            [0, 0, 0, 1]
        ])
    )


def test_inv_motion_matirx():
    R = random_rotation_matrix(3)
    t = np.random.uniform(-1, 1, 3)
    T = motion_matrix(R, t)
    assert_array_almost_equal(inv_motion_matrix(T), inv(T))


def test_calc_relative_transform():
    R_wa = random_rotation_matrix(3)
    t_wa = np.random.uniform(-1, 1, 3)
    T_wa = motion_matrix(R_wa, t_wa)

    R_wb = random_rotation_matrix(3)
    t_wb = np.random.uniform(-1, 1, 3)
    T_wb = motion_matrix(R_wb, t_wb)

    T_ab = calc_relative_transform(T_wa, T_wb)

    assert_array_almost_equal(np.dot(T_wa, T_ab), T_wb)


def test_get_rotation_translation():
    R = random_rotation_matrix(3)
    t = np.random.uniform(-1, 1, 3)

    R_, t_ = get_rotation_translation(motion_matrix(R, t))
    assert_array_equal(R, R_)
    assert_array_equal(t, t_)


def test_estimate_fundamental():
    camera_parameters = CameraParameters(
        focal_length=[0.8, 1.2],
        offset=[0.8, 0.2]
    )
    projection = PerspectiveProjection(camera_parameters)

    R = random_rotation_matrix(3)
    t = np.random.uniform(-10, 10, 3)

    points_true = np.random.uniform(-10, 10, (10, 3))

    keypoints0 = projection.compute(points_true)
    keypoints1 = projection.compute(transform(R, t, points_true))

    K = camera_parameters.matrix
    K_inv = inv(K)

    F = estimate_fundamental(keypoints0, keypoints1)
    E = fundamental_to_essential(F, K)

    for i in range(points_true.shape[0]):
        x0 = np.append(keypoints0[i], 1)
        x1 = np.append(keypoints1[i], 1)
        assert_almost_equal(x1.dot(F).dot(x0), 0)

        y0 = np.dot(K_inv, x0)
        y1 = np.dot(K_inv, x1)
        assert_almost_equal(y1.dot(E).dot(y0), 0)

        # properties of the essential matrix
        assert_almost_equal(np.linalg.det(E), 0)
        assert_array_almost_equal(
            2 * np.dot(E, np.dot(E.T, E)) - np.trace(np.dot(E, E.T)) * E,
            np.zeros((3, 3))
        )


def to_essential(R, t):
    return np.dot(tangent_so3(t), R)


def test_fundamental_to_essential():
    R = random_rotation_matrix(3)
    t = np.random.uniform(-10, 10, 3)
    K0 = CameraParameters(focal_length=[0.8, 1.2], offset=[0.8, -0.2]).matrix
    K1 = CameraParameters(focal_length=[0.7, 0.9], offset=[-1.0, 0.1]).matrix

    E_true = to_essential(R, t)
    F = inv(K1).T.dot(E_true).dot(inv(K0))
    E_pred = fundamental_to_essential(F, K0, K1)
    assert_array_almost_equal(E_true, E_pred)


def test_decompose_essential():
    def test(R_true, t_true):
        # skew matrx corresponding to t
        S_true = tangent_so3(t_true)

        E_true = np.dot(R_true, S_true)

        R1, R2, t1, t2 = decompose_essential(E_true)

        # t1 = -t2, R.T * t1 is parallel to t_true
        assert_array_almost_equal(t1, -t2)
        assert_array_almost_equal(np.cross(np.dot(R1.T, t1), t_true),
                                  np.zeros(3))
        assert_array_almost_equal(np.cross(np.dot(R2.T, t1), t_true),
                                  np.zeros(3))

        # make sure that both of R1 and R2 are rotation matrices
        assert_array_almost_equal(np.dot(R1.T, R1), np.identity(3))
        assert_array_almost_equal(np.dot(R2.T, R2), np.identity(3))
        assert_almost_equal(np.linalg.det(R1), 1.)
        assert_almost_equal(np.linalg.det(R2), 1.)

    N = 10
    angles = np.random.uniform(-np.pi, np.pi, (N, 3))
    rotations = Rotation.from_euler('xyz', angles).as_matrix()
    translations = np.random.uniform(-10, 10, (N, 3))

    for R, t in itertools.product(rotations, translations):
        test(R, t)
