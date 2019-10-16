from autograd import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.rigid_transform import transform
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.keypoints import Matcher
from vitamine.exceptions import NotEnoughInliersException
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.pose import Pose
from vitamine.point_index import PointIndices
from vitamine.rigid_transform import transform_all
from vitamine.utils import random_binary, break_other_than
from tests.data import dummy_points as points_true


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


def test_init_first():
    kd0 = KD(keypoints_true[0], descriptors)
    vo = VisualOdometry(camera_parameters, FOV(0.0), matcher=matcher)
    vo.init_first(kd0, PointIndices(len(keypoints_true[0])))
    assert(vo.keypoint_descriptor_list[0] is kd0)
    assert(vo.poses[0] == Pose.identity())


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

        assert(vo.keypoint_descriptor_list[1] is kd1)
        assert(len(vo.keypoint_descriptor_list) == 2)
        assert(len(vo.poses) == 2)

        indices0, indices1 = vo.point_indices_list
        assert_array_equal(indices0.triangulated, np.arange(len(points_true)))
        assert_array_equal(indices1.triangulated, np.arange(len(points_true)))

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
        # point_indices   0  1  2  3   4   5   6   7
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

        assert(vo.keypoint_descriptor_list[0] is kd0)
        assert(vo.keypoint_descriptor_list[1] is kd2)

        point_indices0, point_indices2 = vo.point_indices_list
        assert_array_equal(point_indices0.is_triangulated,
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        assert_array_equal(point_indices0.triangulated,
                           [0, 1, 2, 3, 4, 5, 6, 7])
        assert_array_equal(point_indices2.is_triangulated,
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        assert_array_equal(point_indices2.triangulated,
                           [0, 1, 2, 3, 4, 5, 6, 7])

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


def assert_projection_equal(points, point_indices, pose, keypoints):
    P = transform(pose.R, pose.t, points.get(point_indices.triangulated))
    assert_array_almost_equal(projection.compute(P),
                              keypoints[point_indices.is_triangulated])


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
    descriptors3 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 8, 9])

    keypoints0, keypoints1, keypoints2 = keypoints_true[0:3]
    kd0 = KD(keypoints0, descriptors0)
    kd1 = KD(keypoints1, descriptors1)
    kd2 = KD(keypoints2, descriptors2)

    assert(vo.try_add_keyframe(kd0))
    assert(vo.n_active_keyframes == 1)

    assert(vo.try_add_keyframe(kd1))
    assert(vo.n_active_keyframes == 2)

    # first triangulation
    # point_indices   0  1  2  3  4   5   6   7   8
    # matches01.T = [[0, 2, 4, 6, 7, 10, 11, 12, 13],
    #                [0, 2, 4, 6, 7, 10, 11, 12, 13]]

    point_indices0, point_indices1 = vo.point_indices_list
    assert_array_equal(point_indices0.is_triangulated,
                       [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    assert_array_equal(point_indices0.triangulated,
                       [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_array_equal(point_indices1.is_triangulated,
                       [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    assert_array_equal(point_indices1.triangulated,
                       [0, 1, 2, 3, 4, 5, 6, 7, 8])

    pose0, pose1 = vo.poses
    assert_projection_equal(vo.points, point_indices0, pose0, keypoints0)
    assert_projection_equal(vo.points, point_indices1, pose1, keypoints1)

    # kd0 and kd1 are already observed and kd2 is added
    # point_indices   2     3  4      5   6   7      existing
    # point_indices      9       10                  newly triangulated
    # matches02.T = [[4, 5, 6, 7, 9, 10, 11, 12],
    #                [4, 5, 6, 7, 9, 10, 11, 12]]
    #
    # point_indices      2  3  4      5   6   7      existing
    # point_indices  11          12                  newly triangulated
    # matches12.T = [[1, 4, 6, 7, 8, 10, 11, 12],
    #                [1, 4, 6, 7, 8, 10, 11, 12]]
    np.set_printoptions(suppress=True)

    assert(vo.try_add_keyframe(kd2))
    assert(vo.n_active_keyframes == 3)

    assert(len(vo.export_points()) == 13)

    point_indices0, point_indices1, point_indices2 = vo.point_indices_list
    assert_array_equal(point_indices0.is_triangulated,
                       [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    assert_array_equal(point_indices0.triangulated,
                       [0, 1, 2, 9, 3, 4, 10, 5, 6, 7, 8])
    assert_array_equal(point_indices1.is_triangulated,
                       [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    assert_array_equal(point_indices1.triangulated,
                       [0, 11, 1, 2, 3, 4, 12, 5, 6, 7, 8])
    # [0, 2, 3, 8] cannot be found in matches01 and matches02
    # so they should not be triangulated
    assert_array_equal(point_indices2.is_triangulated,
                       [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert_array_equal(point_indices2.triangulated,
                       [11, 2, 9, 3, 4, 12, 10, 5, 6, 7])

    pose0, pose1, pose2 = vo.poses
    assert_projection_equal(vo.points, point_indices0, pose0, keypoints0)
    assert_projection_equal(vo.points, point_indices1, pose1, keypoints1)
    assert_projection_equal(vo.points, point_indices2, pose2, keypoints2)


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
