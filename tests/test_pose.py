import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from tadataka.so3 import rodrigues, exp_so3

from tadataka.camera import CameraParameters
from tadataka.exceptions import NotEnoughInliersException
from tadataka.dataset.observations import generate_translations
from tadataka.pose import (
    Pose, solve_pnp, pose_change_from_stereo,
    n_triangulated, triangulation_indices
)
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform

from tests.utils import random_rotation_matrix


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

points = np.array([
   [4, -1, 3],
   [1, -3, -2],
   [-2, 3, -2],
   [-3, -2, -5],
   [4, 1, 1],
   [-2, 3, 1],
   [-4, 4, -1]
])

omegas = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.2, 0.0, 0.1],
    [3.2, 1.1, 1.0],
    [3.1, -0.2, 0.0],
    [-1.0, 0.2, 3.1]
])

translations = generate_translations(rodrigues(omegas), points)


def test_solve_pnp():
    for omega_true, t_true in zip(omegas, translations):
        P = transform(exp_so3(omega_true), t_true, points)
        keypoints_true = projection.compute(P)

        pose = solve_pnp(points, keypoints_true)

        P = transform(pose.R, pose.t, points)
        keypoints_pred = projection.compute(P)

        # omega_true and omega_pred can be different but
        # they have to be linearly dependent and satisfy
        # norm(omega_true) - norm(omega_pred) = 2 * n * pi

        assert_array_almost_equal(t_true, pose.t)
        assert_array_almost_equal(keypoints_true, keypoints_pred)

    P = transform(exp_so3(omegas[0]), translations[0], points)
    keypoints0 = projection.compute(P)

    # 6 correspondences
    # this should be able to run
    solve_pnp(points[0:6], keypoints0[0:6])

    with pytest.raises(NotEnoughInliersException):
        # not enough correspondences
        solve_pnp(points[0:5], keypoints0[0:5])


def test_eq():
    rotaiton0 = Rotation.from_dcm(random_rotation_matrix(3))
    rotaiton1 = Rotation.from_dcm(random_rotation_matrix(3))
    t0 = np.zeros(3)
    t1 = np.arange(3)

    assert(Pose(rotaiton0, t0) == Pose(rotaiton0, t0))
    assert(Pose(rotaiton1, t1) == Pose(rotaiton1, t1))
    assert(Pose(rotaiton0, t0) != Pose(rotaiton0, t1))
    assert(Pose(rotaiton0, t0) != Pose(rotaiton1, t0))
    assert(Pose(rotaiton0, t0) != Pose(rotaiton1, t1))


def test_identity():
    pose = Pose.identity()
    assert_array_equal(pose.rotation.as_rotvec(), np.zeros(3))
    assert_array_equal(pose.t, np.zeros(3))


def test_R():
    pose = Pose(Rotation.from_rotvec(np.zeros(3)), np.zeros(3))
    assert_array_almost_equal(pose.R, np.identity(3))

    rotvec, t = np.array([np.pi, 0, 0]), np.zeros(3)
    pose = Pose(Rotation.from_rotvec(rotvec), t)
    assert_array_almost_equal(pose.R, np.diag([1, -1, -1]))


def test_n_triangulated():
    assert(n_triangulated(1000, 0.2, 40) == 200)   # 1000 * 0.2
    assert(n_triangulated(100, 0.2, 40) == 40)    # max(100 * 0.2, 40)
    assert(n_triangulated(100, 0.2, 800) == 100)  # min(100, 800)


def test_triangulation_indices():
    indices = triangulation_indices(100)
    np.unique(indices) == len(indices)  # non overlapping


def test_estimate_pose_change():
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

    R_true = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    def case1():
        t_true = np.array([0, 0, 5])
        P0 = X_true
        P1 = transform(R_true, t_true, X_true)
        keypoints0 = projection.compute(P0)
        keypoints1 = projection.compute(P1)
        R, t = pose_change_from_stereo(keypoints0, keypoints1)

        assert_array_almost_equal(R, R_true)
        # test if t and t_true are parallel
        # because we cannot know the scale
        assert_array_almost_equal(np.cross(t, t_true), np.zeros(3))

    def case2():
        # 5 points are behind cameras
        t_true = np.array([0, 0, 0])
        P0 = X_true
        P1 = transform(R_true, t_true, X_true)
        keypoints0 = projection.compute(P0)
        keypoints1 = projection.compute(P1)

        message = "Most of points are behind cameras. Maybe wrong matches?"
        with pytest.warns(RuntimeWarning, match=message):
            pose_change_from_stereo(keypoints0, keypoints1)

    case1()
    case2()
