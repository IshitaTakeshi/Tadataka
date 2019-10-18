import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal
from autograd import numpy as np

from vitamine.pose import Pose, solve_pnp
from vitamine.camera import CameraParameters
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.observations import generate_translations
from vitamine.so3 import rodrigues
from vitamine.points import PointManager
from vitamine.rigid_transform import transform
from tests.assertion import assert_projection_equal
from tests.data import dummy_points as points_true


omegas = np.array([
    [0, 0, 0],
    [0, np.pi / 2, 0],
    [np.pi / 2, 0, 0],
    [0, np.pi / 4, 0],
    [0, -np.pi / 4, 0],
    [-np.pi / 4, np.pi / 4, 0],
    [0, np.pi / 8, -np.pi / 4]
])

rotations = rodrigues(omegas)
translations = generate_translations(rotations, points_true)

camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

def test_point_manager():
    point_manager = PointManager()

    # initalization from 0th and 1st view
    P0 = transform(rotations[0], translations[0], points_true)
    P1 = transform(rotations[1], translations[1], points_true)
    P2 = transform(rotations[2], translations[2], points_true)
    keypoints0 = projection.compute(P0)
    keypoints1 = projection.compute(P1)
    keypoints2 = projection.compute(P2)

    matches01 = np.array([[0, 1, 2, 3, 4, 6, 8, 9],
                          [0, 1, 2, 3, 4, 6, 8, 9]]).T

    pose0, pose1 = point_manager.initialize(keypoints0, keypoints1, matches01,
                                            viewpoint0=0, viewpoint1=1)

    assert_array_almost_equal(pose0.omega, np.zeros(3))
    assert_array_almost_equal(pose1.omega, omegas[1])

    P0, mask0 = point_manager.get(0, matches01[:, 0])
    P1, mask1 = point_manager.get(1, matches01[:, 1])
    assert(np.all(mask0))
    assert(np.all(mask1))
    assert_projection_equal(projection, pose0, P0, keypoints0[matches01[:, 0]])
    assert_projection_equal(projection, pose1, P1, keypoints1[matches01[:, 1]])
    assert_array_equal(P0, P1)

    # add the 2nd view

    # triangulate with the 0th view

    # [0, 1, 2, 3, 8, 9] are already triangulated
    # [5, 7] are not triangulated yet
    matches02 = np.array([[0, 1, 2, 3, 5, 7, 8, 9],
                          [0, 1, 2, 3, 5, 7, 8, 9]]).T
    P0, mask0 = point_manager.get(0, matches02[:, 0])
    assert_array_equal(mask0, [1, 1, 1, 1, 0, 0, 1, 1])

    # esitmate the pose by pnp to align the scale
    pose2 = solve_pnp(P0, keypoints2[matches02[mask0, 1]])

    point_manager.triangulate(pose0, pose2, keypoints0, keypoints2, matches02,
                              viewpoint0=0, viewpoint1=2)
    P0, mask0 = point_manager.get(0, matches02[:, 0])
    P2, mask2 = point_manager.get(2, matches02[:, 1])

    # all masks shuld be true because all matched points are triangulated
    assert(np.all(mask0))
    assert(np.all(mask2))

    assert_projection_equal(projection, pose0, P0, keypoints0[matches02[:, 0]])
    assert_projection_equal(projection, pose2, P2, keypoints2[matches02[:, 1]])

    # triangulate with the 1st view

    # [0, 1, 2, 3, 8, 9] are already triangulated in viewpoint 0, 1, 2
    # [4, 6] are triangulated in viewpoint 0, 1
    # [5, 7] are triangulated in viewpoint 0, 2
    # [10, 11, 12] are not triangulated yet in any viewpoints
    matches12 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]).T
    P1, mask1 = point_manager.get(1, matches12[:, 0])
    # keypoint index           0  1  2  3  4  5  6  7  8  9 10 11 12
    assert_array_equal(mask1, [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0])

    # esitmate the pose using matches to the 1st view
    pose2 = solve_pnp(P1, keypoints2[matches12[mask1, 1]])

    point_manager.triangulate(pose1, pose2, keypoints1, keypoints2, matches12,
                              viewpoint0=1, viewpoint1=2)
    P1, mask1 = point_manager.get(1, matches12[:, 0])
    P2, mask2 = point_manager.get(2, matches12[:, 1])

    assert_projection_equal(projection, pose1, P1, keypoints1[matches12[:, 0]])
    assert_projection_equal(projection, pose2, P2, keypoints2[matches12[:, 1]])

    # wrong match
    matches12 = np.array([[0, 1]])
    with pytest.warns(UserWarning):
        point_manager.triangulate(
            pose1, pose2, keypoints1, keypoints2, matches12,
            viewpoint0=1, viewpoint1=2
        )
