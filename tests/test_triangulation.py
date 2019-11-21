import itertools

import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from tadataka.camera import CameraParameters
from tadataka.dataset.observations import generate_translations
from tadataka.matrix import decompose_essential
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform, transform_all
from tadataka.so3 import tangent_so3, rodrigues
from tadataka._triangulation import linear_triangulation, triangulation

# TODO add the case such that x[3] = 0

points_true = np.array([
   [4, -1, 3],
   [1, -3, -2],
   [-2, 3, -2],
   [-3, -2, -5],
   [-3, -1, 2],
   [-4, -2, 3],
   [4, 1, 1],
   [-2, 3, 1],
   [4, 1, 2],
   [-4, 4, -1]
])

omegas = np.array([
    [0, 0, 0],
    [0, 2 * np.pi / 8, 0],
    [0, 4 * np.pi / 8, 0],
    [1 * np.pi / 8, 1 * np.pi / 8, 0],
    [2 * np.pi / 8, 1 * np.pi / 8, 0],
    [1 * np.pi / 8, 1 * np.pi / 8, 1 * np.pi / 8],
])

rotations = rodrigues(omegas)
translations = generate_translations(rotations, points_true)

# rotations = np.array([
#     [[1, 0, 0],
#      [0, 1, 0],
#      [0, 0, 1]],
# ])
#
# translations = np.array([
#     [-8, 4, 8],
#     [4, 8, 9],
#     [4, 1, 7]
# ])


def test_linear_triangulation():
    projection = PerspectiveProjection(
        CameraParameters(focal_length=[1., 1.], offset=[0., 0.])
    )

    t0, t1 = translations[0:2]
    R0, t0 = rotations[0], translations[0]
    R1, t1 = rotations[1], translations[1]

    R0 = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    R1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    keypoints0 = projection.compute(transform(R0, t0, points_true))
    keypoints1 = projection.compute(transform(R1, t1, points_true))

    for i in range(points_true.shape[0]):
        x_true = points_true[i]
        x, depth0, depth1 = linear_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i])
        assert_array_almost_equal(x, x_true)
        assert_equal(depth0, x[1] + t0[2])
        assert_equal(depth1, x[2] + t1[2])


def test_decompose_essential():
    def test(R_true, t_true):
        # skew matrx corresponding to t
        S_true = tangent_so3(t_true.reshape(1, *t_true.shape))[0]

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

    for R, t in itertools.product(rotations, translations):
        test(R, t)


def test_triangulation():
    projection = PerspectiveProjection(
        CameraParameters(focal_length=[1., 1.], offset=[0., 0.])
    )
    X_true = np.array([
        [-1, -6, 5],
        [9, 1, 8],
        [-9, -2, 6],
        [-3, 3, 6],
        [3, -1, 4],
        [-3, 7, -9],
        [7, 1, 4],
        [6, 5, 3],
        [0, -4, 1],
        [9, -1, 7]
    ])

    def run(R1, R2, t1, t2):
        P1 = transform(R1, t1, X_true)
        P2 = transform(R2, t2, X_true)
        depth_mask_true = np.logical_and(P1[:, 2] > 0, P2[:, 2] > 0)
        keypoints1 = projection.compute(P1)
        keypoints2 = projection.compute(P2)
        X_pred, depth_mask = triangulation(R1, R2, t1, t2,
                                           keypoints1, keypoints2)

        assert_array_almost_equal(X_true, X_pred)
        assert_array_equal(depth_mask, depth_mask_true)

    R1 = np.array([[-1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]])
    R2 = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    t1 = np.array([0, 0, 3])
    t2 = np.array([0, 1, 10])

    run(R1, R2, t1, t2)  # 2 points are behind the cameras

    R1 = np.array([[-1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]])
    R2 = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])
    t1 = np.array([3, 0, 2])
    t2 = np.array([1, 1, 6.5])

    message = "Most of points are behind cameras. Maybe wrong matches?"
    with pytest.warns(RuntimeWarning, match=message):
        run(R1, R2, t1, t2)  # 3 points are behind the cameras

    R1 = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                   [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                   [0, 0, 1]])
    R2 = np.array([[-7 / 25, 0, -24 / 25],
                   [0, -1, 0],
                   [-24 / 25, 0, 7 / 25]])
    t1 = np.array([-3.0, 3.0, 10.0])
    t2 = np.array([-1.0, 1.0, 12.5])

    run(R1, R2, t1, t2)  # all points are in front of both cameras
