import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from scipy.spatial.transform import Rotation

from tadataka.camera import CameraParameters
from tadataka.dataset.observations import generate_translations
from tadataka.pose import Pose
from tadataka.projection import PerspectiveProjection, pi
from tadataka.rigid_transform import transform
from tadataka.triangulation import (
    DepthsFromTriangulation, Triangulation, TwoViewTriangulation,
    linear_triangulation, calc_depth0, calc_depth0_)


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
], dtype=np.float64)

R0 = Rotation.from_euler('xyz', np.random.random(3)).as_matrix()
R1 = Rotation.from_euler('xyz', np.random.random(3)).as_matrix()
R2 = Rotation.from_euler('xyz', np.random.random(3)).as_matrix()

[t0, t1, t2] = generate_translations(
    np.array([R0, R1, R2]), points_true
)

keypoints0 = pi(transform(R0, t0, points_true))
keypoints1 = pi(transform(R1, t1, points_true))
keypoints2 = pi(transform(R2, t2, points_true))


def test_linear_triangulation():
    rotations = np.array([R0, R1, R2])
    translations = np.array([t0, t1, t2])
    keypoints = np.stack((
        keypoints0,
        keypoints1,
        keypoints2
    ))

    points, depths = linear_triangulation(rotations, translations, keypoints)

    assert_array_almost_equal(points, points_true)

    assert(depths.shape == (3, points_true.shape[0]))
    for i, x in enumerate(points_true):
        assert_array_almost_equal(
            depths[:, i],
            [np.dot(R0, x)[2] + t0[2],
             np.dot(R1, x)[2] + t1[2],
             np.dot(R2, x)[2] + t2[2]]
        )


def test_two_view_triangulation():
    triangulator = TwoViewTriangulation(
        Pose(Rotation.from_matrix(R0), t0),
        Pose(Rotation.from_matrix(R1), t1)
    )

    points, depths = triangulator.triangulate(keypoints0, keypoints1)

    assert_array_almost_equal(points, points_true)

    assert(depths.shape == (2, points_true.shape[0]))

    for i, x in enumerate(points_true):
        assert_array_almost_equal(
            depths[:, i],
            [np.dot(R0, x)[2] + t0[2],
             np.dot(R1, x)[2] + t1[2]]
        )


def test_triangulation():
    triangulator = Triangulation(
        [Pose(Rotation.from_matrix(R0), t0),
         Pose(Rotation.from_matrix(R1), t1),
         Pose(Rotation.from_matrix(R2), t2)]
    )

    keypoints = np.stack((
        keypoints0,
        keypoints1,
        keypoints2
    ))

    points, depths = triangulator.triangulate(keypoints)

    assert_array_almost_equal(points, points_true)

    assert(depths.shape == (3, points_true.shape[0]))

    for i, x in enumerate(points_true):
        assert_array_almost_equal(
            depths[:, i],
            [np.dot(R0, x)[2] + t0[2],  # dot(R0, x)[2] + t0[2]
             np.dot(R1, x)[2] + t1[2],  # dot(R1, x)[2] + t1[2]
             np.dot(R2, x)[2] + t2[2]]  # dot(R2, x)[2] + t2[2]
        )


def test_depths_from_triangulation():
    rotation0 = Rotation.from_quat([0, 0, 0, 1])
    rotation1 = Rotation.from_quat([0, 0, 1, 0])
    t0 = np.array([-1, 3, 4], dtype=np.float64)
    t1 = np.array([4, 1, 6], dtype=np.float64)
    point = np.array([0, 0, 5], dtype=np.float64)

    p0 = transform(rotation0.as_matrix(), t0, point)
    p1 = transform(rotation1.as_matrix(), t1, point)
    x0 = pi(p0)
    x1 = pi(p1)

    pose0 = Pose(rotation0, t0)
    pose1 = Pose(rotation1, t1)
    depths = DepthsFromTriangulation(pose0, pose1)(x0, x1)
    assert_array_almost_equal(depths, [p0[2], p1[2]])


def test_calc_depth0_():
    point = np.array([0, 0, 5], dtype=np.float64)

    def run(pose0w, pose1w):
        p0 = transform(pose0w.R, pose0w.t, point)
        p1 = transform(pose1w.R, pose1w.t, point)
        x0 = pi(p0)
        x1 = pi(p1)

        pose10 = pose1w * pose0w.inv()
        depth = calc_depth0_(pose10.R, pose10.t, x0, x1)
        assert_array_almost_equal(depth, p0[2])

    rotation1 = Rotation.from_rotvec(np.random.uniform(-np.pi, np.pi, 3))
    t1 = np.array([4, 1, 6])

    pose0w = Pose.identity()
    pose1w = Pose(rotation1, t1)
    run(pose0w, pose1w)

    rotation = Rotation.identity()
    pose0w = Pose(rotation, np.array([-5, 0, 0]))
    pose1w = Pose(rotation, np.array([5, 0, 0]))
    run(pose0w, pose1w)


def test_calc_depth0():
    rotation0 = Rotation.from_rotvec([0, np.pi/2, 0])
    t0 = np.array([-3, 0, 1])

    rotation1 = Rotation.from_rotvec([0, -np.pi/2, 0])
    t1 = np.array([0, 0, 2])

    pose_w0 = Pose(rotation0, t0)
    pose_w1 = Pose(rotation1, t1)

    point = np.array([-1, 0, 1], dtype=np.float64)

    pose_0w = pose_w0.inv()
    pose_1w = pose_w1.inv()

    x0 = pi(transform(pose_0w.R, pose_0w.t, point))
    x1 = pi(transform(pose_1w.R, pose_1w.t, point))
    assert_almost_equal(calc_depth0(pose_w0, pose_w1, x0, x1), 2)
