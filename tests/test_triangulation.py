import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from scipy.spatial.transform import Rotation

from tadataka.camera import CameraParameters
from tadataka.dataset.observations import generate_translations
from tadataka.pose import Pose
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform
from tadataka._triangulation import linear_triangulation
from tadataka.triangulation import Triangulation, TwoViewTriangulation

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

projection = PerspectiveProjection(
    CameraParameters(focal_length=[1., 1.], offset=[0., 0.])
)

R0 = Rotation.from_euler('xyz', np.random.random(3)).as_dcm()
R1 = Rotation.from_euler('xyz', np.random.random(3)).as_dcm()
R2 = Rotation.from_euler('xyz', np.random.random(3)).as_dcm()

[t0, t1, t2] = generate_translations(
    np.array([R0, R1, R2]), points_true
)

keypoints0 = projection.compute(transform(R0, t0, points_true))
keypoints1 = projection.compute(transform(R1, t1, points_true))
keypoints2 = projection.compute(transform(R2, t2, points_true))


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
        Pose(Rotation.from_dcm(R0), t0),
        Pose(Rotation.from_dcm(R1), t1)
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
        [Pose(Rotation.from_dcm(R0), t0),
         Pose(Rotation.from_dcm(R1), t1),
         Pose(Rotation.from_dcm(R2), t2)]
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
