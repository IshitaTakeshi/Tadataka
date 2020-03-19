import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from tadataka.so3 import exp_so3
from tadataka.camera import CameraParameters
from tadataka.exceptions import NotEnoughInliersException
from tadataka.dataset.observations import generate_translations
from tadataka.pose import (
    calc_relative_pose, estimate_pose_change, n_triangulated,
    solve_pnp, triangulation_indices, _Pose, LocalPose, WorldPose
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
], dtype=np.float64)

omegas = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.2, 0.0, 0.1],
    [3.2, 1.1, 1.0],
    [3.1, -0.2, 0.0],
    [-1.0, 0.2, 3.1]
])

translations = generate_translations(exp_so3(omegas), points)


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
    rotaiton0 = Rotation.from_matrix(random_rotation_matrix(3))
    rotaiton1 = Rotation.from_matrix(random_rotation_matrix(3))
    t0 = np.zeros(3)
    t1 = np.arange(3)

    assert(_Pose(rotaiton0, t0) == _Pose(rotaiton0, t0))
    assert(_Pose(rotaiton1, t1) == _Pose(rotaiton1, t1))
    assert(_Pose(rotaiton0, t0) != _Pose(rotaiton0, t1))
    assert(_Pose(rotaiton0, t0) != _Pose(rotaiton1, t0))
    assert(_Pose(rotaiton0, t0) != _Pose(rotaiton1, t1))


def test_identity():
    pose = _Pose.identity()
    assert_array_equal(pose.rotation.as_rotvec(), np.zeros(3))
    assert_array_equal(pose.t, np.zeros(3))


def test_R():
    pose = _Pose(Rotation.from_rotvec(np.zeros(3)), np.zeros(3))
    assert_array_almost_equal(pose.R, np.identity(3))

    rotvec, t = np.array([np.pi, 0, 0]), np.zeros(3)
    pose = _Pose(Rotation.from_rotvec(rotvec), t)
    assert_array_almost_equal(pose.R, np.diag([1, -1, -1]))


def test_inv():
    for i in range(10):
        rotvec = np.random.uniform(-np.pi, np.pi, 3)
        rotation = Rotation.from_rotvec(rotvec)

        t = np.random.uniform(-10, 10, 3)

        p = _Pose(rotation, t)

        q = p * p.inv()
        assert_array_almost_equal(q.rotation.as_rotvec(), np.zeros(3))
        assert_array_almost_equal(q.t, np.zeros(3))


def test_mul():
    # case1
    pose1 = _Pose(Rotation.from_rotvec(np.zeros(3)), np.ones(3))
    pose2 = _Pose(Rotation.from_rotvec(np.zeros(3)), np.ones(3))
    pose3 = pose1 * pose2
    assert_array_equal(pose3.rotation.as_rotvec(), np.zeros(3))
    assert_array_equal(pose3.t, 2 * np.ones(3))

    # case2
    axis = np.array([0.0, 1.0, 2.0])
    rotvec1 = 0.1 * axis
    rotvec2 = 0.4 * axis
    t1 = np.array([0.2, 0.4, -0.1])
    t2 = np.array([-0.1, 2.0, 0.1])
    pose1 = _Pose(Rotation.from_rotvec(rotvec1), t1)
    pose2 = _Pose(Rotation.from_rotvec(rotvec2), t2)
    pose3 = pose1 * pose2

    assert_array_almost_equal(pose3.rotation.as_rotvec(), 0.5 * axis)
    R1 = pose1.rotation.as_matrix()
    assert_array_almost_equal(pose3.t, np.dot(R1, pose2.t) + pose1.t)

    # case3
    # point in the world coordinate system
    point = np.array([0, 0, 1], dtype=np.float64)

    # rotate 90 degrees around the y-axis and move 1 along the z-axis
    rotvec1 = [0, np.pi / 2, 0]
    t1 = np.array([0, 0, 1])

    # rotate 180 degrees around the z-axis and move 2 along the y axis
    rotvec2 = [0, 0, np.pi]
    t2 = np.array([0, 2, 0])

    pose1 = WorldPose(Rotation.from_rotvec(rotvec1), t1)
    pose2 = WorldPose(Rotation.from_rotvec(rotvec2), t2)
    pose3 = pose2 * pose1

    expected = [-1, 2, 1]
    assert_array_almost_equal(
        transform(pose2.R, pose2.t, transform(pose1.R, pose1.t, point)),
        expected
    )
    assert_array_almost_equal(
        transform(pose3.R, pose3.t, point),
        expected
    )


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
    ], dtype=np.float64)

    R_true = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=np.float64)

    def case1():
        t_true = np.array([0, 0, 5], dtype=np.float64)
        P0 = X_true
        P1 = transform(R_true, t_true, X_true)
        keypoints0 = projection.compute(P0)
        keypoints1 = projection.compute(P1)
        pose = estimate_pose_change(keypoints0, keypoints1)

        assert_array_almost_equal(pose.R, R_true)
        # test if t and t_true are parallel
        # because we cannot know the scale
        assert_array_almost_equal(np.cross(pose.t, t_true), np.zeros(3))

    def case2():
        # 5 points are behind cameras
        t_true = np.array([0, 0, 0], dtype=np.float64)
        P0 = X_true
        P1 = transform(R_true, t_true, X_true)
        keypoints0 = projection.compute(P0)
        keypoints1 = projection.compute(P1)

        message = "Most of points are behind cameras. Maybe wrong matches?"
        with pytest.warns(RuntimeWarning, match=message):
            estimate_pose_change(keypoints0, keypoints1)

    case1()
    case2()


def test_calc_relative_pose():
    rotation0 = Rotation.from_rotvec([0.3, -0.1, 0.2])
    rotation1 = Rotation.from_rotvec([-0.1, -1.1, 1.4])
    t0 = np.array([3.0, 0.0, 2.1])
    t1 = np.array([-4.3, 6.1, -1.1])

    point = np.array([4.0, 2.0, -5.0])  # point in the world coordinate

    def case_local():
        # pose0 represents transformation from the world coordinate
        # onto the 0th camera coordinate
        # pose1 represents transformation from the world coordinate
        # onto the 1st camera coordinate
        # pose01 should be the transformation
        # from the 0th camera coordinate
        # onto the 1st camera coordinate
        pose0 = LocalPose(rotation0, t0)
        pose1 = LocalPose(rotation1, t1)

        pose01 = calc_relative_pose(pose0, pose1)

        p0 = transform(pose0.R, pose0.t, point)
        p1_pred = transform(pose01.R, pose01.t, p0)

        p1_true = transform(pose1.R, pose1.t, point)

        assert_array_almost_equal(p1_true, p1_pred)

    def case_world():
        # pose0 transforms a point in the world coordinate
        # to another location (say this is the 0th location) in the world
        # pose1 transforms a point in the world coordinate
        # to another location (say this is the 1st location) in the world
        # pose01 should be the transformation
        # from the 0th location to the 1st location
        pose0 = WorldPose(rotation0, t0)
        pose1 = WorldPose(rotation1, t1)

        pose01 = calc_relative_pose(pose0, pose1)

        p0 = transform(pose0.R, pose0.t, point)
        p1_pred = transform(pose01.R, pose01.t, p0)

        p1_true = transform(pose1.R, pose1.t, point)

        assert_array_almost_equal(p1_true, p1_pred)

    case_local()
    case_world()


def test_to_local():
    rotation = Rotation.from_matrix([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    t = np.array([1, 0, 0])
    local_pose = WorldPose(rotation, t).to_local()
    assert(isinstance(local_pose, LocalPose))

    assert_array_almost_equal(
        local_pose.R,
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    )
    assert_array_almost_equal(local_pose.t, [0, 0, -1])


def test_to_world():
    rotation = Rotation.from_matrix([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    t = np.array([0, 0, -1])
    world_pose = LocalPose(rotation, t).to_world()

    assert(isinstance(world_pose, WorldPose))

    assert_array_almost_equal(
        world_pose.R,
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]]
    )
    assert_array_almost_equal(world_pose.t, [1, 0, 0])


def test_type():
    assert(isinstance(LocalPose.identity(), LocalPose))
    assert(isinstance(WorldPose.identity(), WorldPose))

    xi = np.random.random(6)
    assert(isinstance(LocalPose.from_se3(xi), LocalPose))
    assert(isinstance(WorldPose.from_se3(xi), WorldPose))

    rotvec = Rotation.from_rotvec(np.random.random(3))
    t = np.random.random(3)
    assert(isinstance(LocalPose(rotvec, t).inv(), LocalPose))
    assert(isinstance(WorldPose(rotvec, t).inv(), WorldPose))
    assert(isinstance(LocalPose(rotvec, t) * LocalPose(rotvec, t), LocalPose))
    assert(isinstance(WorldPose(rotvec, t) * WorldPose(rotvec, t), WorldPose))

    message = "Types do not match: LocalPose and WorldPose"
    with pytest.raises(ValueError, match=message):
        LocalPose(rotvec, t) * WorldPose(rotvec, t)

    with pytest.raises(ValueError, match=message):
        LocalPose(rotvec, t) == WorldPose(rotvec, t)

    message = "Types do not match: WorldPose and LocalPose"
    with pytest.raises(ValueError, match=message):
        WorldPose(rotvec, t) * LocalPose(rotvec, t)

    with pytest.raises(ValueError, match=message):
        WorldPose(rotvec, t) == LocalPose(rotvec, t)
