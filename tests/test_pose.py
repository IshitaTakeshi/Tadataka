import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from tadataka.projection import pi
from tadataka.so3 import exp_so3
from tadataka.camera import CameraParameters
from tadataka.exceptions import NotEnoughInliersException
from tadataka.dataset.observations import generate_translations
from tadataka.pose import (
    estimate_pose_change, n_triangulated,
    solve_pnp, triangulation_indices, Pose
)
from tadataka.rigid_transform import transform

from tests.utils import random_rotation_matrix


def test_solve_pnp():
    points = np.random.uniform(-10, 10, (7, 3))
    omegas = np.random.uniform(-np.pi, np.pi, (6, 3))

    translations = generate_translations(exp_so3(omegas), points)

    for omega_true, t_true in zip(omegas, translations):
        P = transform(exp_so3(omega_true), t_true, points)
        keypoints_true = pi(P)

        pose = solve_pnp(points, keypoints_true)

        P = transform(pose.R, pose.t, points)
        keypoints_pred = pi(P)

        # omega_true and omega_pred can be different but
        # they have to be linearly dependent and satisfy
        # norm(omega_true) - norm(omega_pred) = 2 * n * pi

        assert_array_almost_equal(t_true, pose.t)
        assert_array_almost_equal(keypoints_true, keypoints_pred)

    P = transform(exp_so3(omegas[0]), translations[0], points)
    keypoints0 = pi(P)

    # 6 correspondences
    # this should be able to run
    solve_pnp(points[0:6], keypoints0[0:6])

    with pytest.raises(NotEnoughInliersException):
        # not enough correspondences
        solve_pnp(points[0:5], keypoints0[0:5])


def test_eq():
    rotaiton0 = Rotation.from_matrix(random_rotation_matrix(3))
    rotaiton1 = Rotation.from_matrix(random_rotation_matrix(3))
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


def test_T():
    rotvec = np.array([0, np.pi/2, 0])
    t = np.array([-3, -2, 3])
    pose = Pose(Rotation.from_rotvec(rotvec), t)
    assert_array_almost_equal(
        pose.T,
        [[0, 0, 1, -3],
         [0, 1, 0, -2],
         [-1, 0, 0, 3],
         [0, 0, 0, 1]]
    )


def test_inv():
    for i in range(10):
        rotvec = np.random.uniform(-np.pi, np.pi, 3)
        rotation = Rotation.from_rotvec(rotvec)

        t = np.random.uniform(-10, 10, 3)

        p = Pose(rotation, t)

        q = p * p.inv()
        assert_array_almost_equal(q.rotation.as_rotvec(), np.zeros(3))
        assert_array_almost_equal(q.t, np.zeros(3))


def test_mul():
    # case1
    pose1 = Pose(Rotation.from_rotvec(np.zeros(3)), np.ones(3))
    pose2 = Pose(Rotation.from_rotvec(np.zeros(3)), np.ones(3))
    pose3 = pose1 * pose2
    assert_array_equal(pose3.rotation.as_rotvec(), np.zeros(3))
    assert_array_equal(pose3.t, 2 * np.ones(3))

    # case2
    axis = np.array([0.0, 1.0, 2.0])
    rotvec10 = 0.4 * axis
    rotvec21 = 0.1 * axis
    t10 = np.array([-0.1, 2.0, 0.1])
    t21 = np.array([0.2, 0.4, -0.1])
    pose10 = Pose(Rotation.from_rotvec(rotvec10), t10)
    pose21 = Pose(Rotation.from_rotvec(rotvec21), t21)
    pose20 = pose21 * pose10

    assert_array_almost_equal(pose20.rotation.as_rotvec(), 0.5 * axis)
    assert_array_almost_equal(pose20.t, np.dot(pose21.R, pose10.t) + pose21.t)

    # case3
    point = np.random.random(3)

    rotvec21 = np.random.random(3)
    t21 = np.random.uniform(-10, 10, 3)

    rotvec10 = np.random.random(3)
    t10 = np.random.uniform(-10, 10, 3)

    pose10 = Pose(Rotation.from_rotvec(rotvec10), t10)
    pose21 = Pose(Rotation.from_rotvec(rotvec21), t21)
    pose20 = pose21 * pose10

    assert_array_almost_equal(
        transform(pose21.R, pose21.t, transform(pose10.R, pose10.t, point)),
        transform(pose20.R, pose20.t, point)
    )


def test_n_triangulated():
    assert(n_triangulated(1000, 0.2, 40) == 200)   # 1000 * 0.2
    assert(n_triangulated(100, 0.2, 40) == 40)    # max(100 * 0.2, 40)
    assert(n_triangulated(100, 0.2, 800) == 100)  # min(100, 800)


def test_triangulation_indices():
    indices = triangulation_indices(100)
    np.unique(indices) == len(indices)  # non overlapping


def test_estimate_pose_change():
    X0 = np.array([
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
    ], dtype=np.float64)

    R10 = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=np.float64)

    def case1():
        t10 = np.array([0, 0, 5], dtype=np.float64)
        P0 = X0
        P1 = transform(R10, t10, X0)
        keypoints0 = pi(P0)
        keypoints1 = pi(P1)
        pose10 = estimate_pose_change(keypoints0, keypoints1)

        assert_array_almost_equal(pose10.R, R10)
        # test if t pred and t true are parallel
        # because we cannot know the scale
        assert_array_almost_equal(np.cross(pose10.t, t10), np.zeros(3))

    def case2():
        # 5 points are behind cameras
        t10 = np.array([0, 0, 0], dtype=np.float64)
        P0 = X0
        P1 = transform(R10, t10, X0)
        keypoints0 = pi(P0)
        keypoints1 = pi(P1)

        message = "Most of points are behind cameras. Maybe wrong matches?"
        with pytest.warns(RuntimeWarning, match=message):
            estimate_pose_change(keypoints0, keypoints1)

    case1()
    case2()


def test_type():
    assert(isinstance(Pose.identity(), Pose))
    assert(isinstance(Pose.identity(), Pose))

    xi = np.random.random(6)
    assert(isinstance(Pose.from_se3(xi), Pose))
    assert(isinstance(Pose.from_se3(xi), Pose))

    rotvec = Rotation.from_rotvec(np.random.random(3))
    t = np.random.random(3)
    assert(isinstance(Pose(rotvec, t).inv(), Pose))
    assert(isinstance(Pose(rotvec, t).inv(), Pose))
    assert(isinstance(Pose(rotvec, t) * Pose(rotvec, t), Pose))
    assert(isinstance(Pose(rotvec, t) * Pose(rotvec, t), Pose))
