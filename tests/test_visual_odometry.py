from autograd import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from tadataka.so3 import rodrigues
from tadataka.projection import PerspectiveProjection
from tadataka.dataset.points import cubic_lattice
from tadataka.dataset.observations import (
    generate_observations, generate_translations)
from tadataka.rigid_transform import transform
from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV
from tadataka.features import Features as KD
from tadataka.features import Matcher
from tadataka.exceptions import NotEnoughInliersException
from tadataka.visual_odometry import VisualOdometry
from tadataka.pose import Pose
from tadataka.rigid_transform import transform_all
from tadataka.utils import random_binary, break_other_than
from tests.data import dummy_points as points_true
from tests.assertion import assert_projection_equal


matcher = Matcher(enable_ransac=False, enable_homography_filter=False)

camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

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
keypoints_true, positive_depth_mask = generate_observations(
    rotations, translations, points_true, projection
)

# generate dummy descriptors
# allocate sufficient lengths of descriptors for redundancy
descriptors = random_binary((len(points_true), 1024))


def test_try_init_second():
    def case1():
        keypoints_true0 = keypoints_true[0]
        keypoints_true1 = keypoints_true[1]

        kd0 = KD(keypoints_true0, descriptors)
        kd1 = KD(keypoints_true1, descriptors)

        vo = VisualOdometry(camera_parameters, FOV(0.0), matcher=matcher)
        assert(vo.try_add_keyframe(kd0))
        assert(vo.try_add_keyframe(kd1))

        pose0, pose1 = vo.poses
        points_pred = vo.export_points()

        P0 = transform(pose0.R, pose0.t, points_pred)
        P1 = transform(pose1.R, pose1.t, points_pred)
        keypoints_pred0 = projection.compute(P0)
        keypoints_pred1 = projection.compute(P1)

        assert_array_almost_equal(keypoints_true0, keypoints_pred0)
        assert_array_almost_equal(keypoints_true1, keypoints_pred1)

        assert(vo.kds[1] is kd1)
        assert(len(vo.kds) == 2)
        assert(len(vo.poses) == 2)

    def case2():
        # descriptors[0:4] cannot match
        descriptors0 = break_other_than(
            descriptors,
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        )
        # descriptors[4:7] cannot match
        descriptors1 = break_other_than(
            descriptors,
            [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13]
        )
        # descriptors[4:6] cannot match
        descriptors2 = break_other_than(
            descriptors,
            [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        )
        # matches02.T = [[6, 7, 8, 9, 10, 11, 12, 13],
        #                [6, 7, 8, 9, 10, 11, 12, 13]]
        kd0 = KD(keypoints_true[0], descriptors0)
        kd1 = KD(keypoints_true[1], descriptors1)
        kd2 = KD(keypoints_true[2], descriptors2)

        vo = VisualOdometry(camera_parameters, FOV(0.0),
                            matcher=matcher, min_matches=8)
        assert(vo.try_add_keyframe(kd0))
        # number of matches = 7
        # not enough matches found between descriptors0 and descriptors1
        assert(not vo.try_add_keyframe(kd1))
        # number of matches = 8
        # enough matches can be found
        assert(vo.try_add_keyframe(kd2))

        assert(vo.kds[0] is kd0)
        assert(vo.kds[1] is kd2)

        points_pred = vo.export_points()
        pose0, pose2 = vo.poses
        P0 = transform(pose0.R, pose0.t, points_pred)
        P2 = transform(pose2.R, pose2.t, points_pred)
        keypoints_pred0 = projection.compute(P0)
        keypoints_pred2 = projection.compute(P2)

        assert_array_almost_equal(keypoints_true[0, 6:14], keypoints_pred0)
        assert_array_almost_equal(keypoints_true[2, 6:14], keypoints_pred2)

    case1()
    case2()


def test_try_add_more():
    # TODO test with keypoints of different lengths
    vo = VisualOdometry(camera_parameters, FOV(0.0),
                        matcher=matcher, min_matches=8)

    descriptors0 = break_other_than(descriptors,
                                    [0, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13])
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    keypoints0, keypoints1, keypoints2 = keypoints_true[0:3]
    kd0 = KD(keypoints0, descriptors0)
    kd1 = KD(keypoints1, descriptors1)
    kd2 = KD(keypoints2, descriptors2)

    assert(vo.try_add_keyframe(kd0))
    assert(vo.n_active_keyframes == 1)

    assert(vo.try_add_keyframe(kd1))
    assert(vo.n_active_keyframes == 2)

    # first triangulation
    # point indices        0  1  2  3  4   5   6   7   8
    matches01 = np.array([[0, 2, 4, 6, 7, 10, 11, 12, 13],
                          [0, 2, 4, 6, 7, 10, 11, 12, 13]]).T

    print(omegas[2])
    indices0, indices1 = matches01[:, 0], matches01[:, 1]
    pose0, pose1 = vo.poses
    points0 = np.array([vo.point_manager.get(0, i) for i in indices0])
    points1 = np.array([vo.point_manager.get(1, i) for i in indices1])
    assert_projection_equal(projection, pose0, points0, keypoints0[indices0])
    assert_projection_equal(projection, pose1, points1, keypoints1[indices1])

    # kd0 and kd1 are already observed and kd2 is added
    # point indices        2     3  4      5   6   7      existing
    # point indices           9       10                  newly triangulated
    matches02 = np.array([[4, 5, 6, 7, 9, 10, 11, 12],
                          [4, 5, 6, 7, 9, 10, 11, 12]]).T
    #
    # point indices           2  3  4      5   6   7      existing
    # point indices       11          12                  newly triangulated
    matches12 = np.array([[1, 4, 6, 7, 8, 10, 11, 12],
                          [1, 4, 6, 7, 8, 10, 11, 12]]).T

    assert(vo.try_add_keyframe(kd2))
    assert(vo.n_active_keyframes == 3)

    assert(len(vo.export_points()) == 13)

    pose0, pose1, pose2 = vo.poses

    indices0, indices2 = matches02[:, 0], matches02[:, 1]
    points0 = np.array([vo.point_manager.get(0, i) for i in indices0])
    points2 = np.array([vo.point_manager.get(2, i) for i in indices2])
    assert_projection_equal(projection, pose0, points0, keypoints0[indices0])
    assert_projection_equal(projection, pose2, points2, keypoints2[indices2])

    indices1, indices2 = matches12[:, 0], matches12[:, 1]
    points1 = np.array([vo.point_manager.get(1, i) for i in indices1])
    points2 = np.array([vo.point_manager.get(2, i) for i in indices2])
    assert_projection_equal(projection, pose1, points1, keypoints1[indices1])
    assert_projection_equal(projection, pose2, points2, keypoints2[indices2])


def test_try_remove():
    vo = VisualOdometry(camera_parameters, FOV(0.0),
                        matcher=matcher, min_active_keyframes=4)

    assert(vo.try_add(KD(keypoints_true[0], descriptors)))
    assert(vo.n_active_keyframes == 1)

    assert(vo.try_add(KD(keypoints_true[1], descriptors)))
    assert(vo.n_active_keyframes == 2)

    assert(vo.try_add(KD(keypoints_true[2], descriptors)))
    assert(vo.n_active_keyframes == 3)

    assert(vo.try_add(KD(keypoints_true[3], descriptors)))
    assert(vo.n_active_keyframes == 4)

    assert(not vo.try_remove())
    assert(vo.n_active_keyframes == 4)

    assert(vo.try_add(KD(keypoints_true[4], descriptors)))
    assert(vo.n_active_keyframes == 5)

    assert(vo.try_remove())
    assert(vo.n_active_keyframes == 4)

    assert(not vo.try_remove())
    assert(vo.n_active_keyframes == 4)
