from autograd import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.rigid_transform import transform
from vitamine.visual_odometry import visual_odometry
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.keypoints import match
from vitamine.exceptions import NotEnoughInliersException
from vitamine.visual_odometry.visual_odometry import (
    VisualOdometry, find_best_match, estimate_pose,
    get_correspondences)
from vitamine.pose import Pose
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keypoint import LocalFeatures
from vitamine.rigid_transform import transform_all
from vitamine.utils import random_binary, break_other_than
from tests.data import dummy_points as points_true


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

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
keypoints_true, positive_depth_mask = generate_observations(
    rotations, translations, points_true, projection
)

# generate dummy descriptors
# allocate sufficient lengths of descriptors for redundancy
descriptors = random_binary((len(points_true), 1024))


def test_init_first():
    lf0 = LocalFeatures(keypoints_true[0], descriptors)
    vo = VisualOdometry(camera_parameters, FOV(0.0))
    vo.init_first(lf0)
    assert(vo.local_features[0] is lf0)
    assert(vo.poses[0] == Pose.identity())



def test_try_init_second():
    def case1():
        keypoints_true0 = keypoints_true[0]
        keypoints_true1 = keypoints_true[1]

        lf0 = LocalFeatures(keypoints_true0, descriptors)
        lf1 = LocalFeatures(keypoints_true1, descriptors)

        vo = VisualOdometry(camera_parameters, FOV(0.0))
        assert(vo.try_add_keyframe(lf0))
        assert(vo.try_add_keyframe(lf1))

        pose0, pose1 = vo.poses
        points_pred = vo.export_points()

        P0 = transform(pose0.R, pose0.t, points_pred)
        P1 = transform(pose1.R, pose1.t, points_pred)
        keypoints_pred0 = projection.compute(P0)
        keypoints_pred1 = projection.compute(P1)

        assert_array_almost_equal(keypoints_true0, keypoints_pred0)
        assert_array_almost_equal(keypoints_true1, keypoints_pred1)

        assert(vo.local_features[1] is lf1)
        assert(len(vo.local_features) == 2)
        assert(len(vo.poses) == 2)

        lf0, lf1 = vo.local_features
        assert_array_equal(lf0.point_indices, np.arange(len(points_true)))
        assert_array_equal(lf1.point_indices, np.arange(len(points_true)))

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
        lf0 = LocalFeatures(keypoints_true[0], descriptors0)
        lf1 = LocalFeatures(keypoints_true[1], descriptors1)
        lf2 = LocalFeatures(keypoints_true[2], descriptors2)

        vo = VisualOdometry(camera_parameters, FOV(0.0), min_matches=8)
        assert(vo.try_add_keyframe(lf0))
        # number of matches = 7
        # not enough matches found between descriptors0 and descriptors1
        assert(not vo.try_add_keyframe(lf1))
        # number of matches = 8
        # enough matches can be found
        assert(vo.try_add_keyframe(lf2))

        assert(vo.local_features[0] is lf0)
        assert(vo.local_features[1] is lf2)

        assert_array_equal(lf0.point_indices,
                           [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7])
        assert_array_equal(lf2.point_indices,
                           [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7])

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


def test_get_correspondences():
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [0, 1, 4, 5, 6, 7, 9, 10, 11, 12, 13])
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)
    lf1.point_indices = np.array(
        # 0   1  2  3  4   5   6   7   8  9  10  11 12 13
        [-1, -1, 4, 5, 0, -1, -1, -1, -1, 1, -1, -1, 2, 3]
    )
    lf2.point_indices = np.array(
        # 0   1   2   3  4  5  6  7   8  9  10  11 12 13
        [-1, -1, -1, -1, 0, 6, 7, 8, -1, 1, -1, -1, 2, 3]
    )

    def case1():
        descriptors0 = break_other_than(descriptors,
                                        [2, 3, 4, 5, 6, 7, 9, 12, 13])
        #                 4  5  0  1   2   3
        # matches01.T = [[2, 3, 4, 9, 12, 13],
        #                [2, 3, 4, 9, 12, 13]]
        #                 0  6  7  8  1   2   3
        # matches02.T = [[4, 5, 6, 7, 9, 12, 13],
        #                [4, 5, 6, 7, 9, 12, 13]]

        keypoints0 = keypoints_true[0]
        lf0 = LocalFeatures(keypoints0, descriptors0)

        point_indices, keypoints = get_correspondences(match, [lf1, lf2], lf0)

        assert_array_equal(point_indices,
                           [4, 5, 0, 1, 2, 3, 0, 6, 7, 8, 1, 2, 3])
        expected = np.vstack([
            keypoints0[2], keypoints0[3], keypoints0[4],
            keypoints0[9], keypoints0[12], keypoints0[13],
            keypoints0[4], keypoints0[5], keypoints0[6],
            keypoints0[7], keypoints0[9], keypoints0[12], keypoints0[13]
        ])
        assert_array_equal(keypoints, expected)

    def case2():
        descriptors0 = break_other_than(descriptors[2:14],
                                        [0, 1, 2, 3, 4, 5, 7, 10, 11])
        #                 4  5  0  1   2   3
        # matches01.T = [[0, 1, 2, 7, 10, 11],
        #                [2, 3, 4, 9, 12, 13]]
        #                 0  6  7  8  1   2   3
        # matches02.T = [[2, 3, 4, 5, 7, 10, 11],
        #                [4, 5, 6, 7, 9, 12, 13]]
        keypoints0 = keypoints_true[0, 2:14]
        lf0 = LocalFeatures(keypoints0, descriptors0)

        point_indices, keypoints = get_correspondences(match, [lf1, lf2], lf0)

        assert_array_equal(point_indices,
                           [4, 5, 0, 1, 2, 3, 0, 6, 7, 8, 1, 2, 3])
        expected = np.vstack([
            keypoints0[0], keypoints0[1], keypoints0[2],
            keypoints0[7], keypoints0[10], keypoints0[11],
            keypoints0[2], keypoints0[3], keypoints0[4],
            keypoints0[5], keypoints0[7], keypoints0[10], keypoints0[11]
        ])
        assert_array_equal(keypoints, expected)

    def case3():
        # none of them can match
        descriptors0 = descriptors
        descriptors1 = random_binary((10, descriptors.shape[1]))
        descriptors2 = random_binary((12, descriptors.shape[1]))
        lf0 = LocalFeatures(keypoints_true[0], descriptors0)
        lf1 = LocalFeatures(keypoints_true[1, 0:10], descriptors1)
        lf2 = LocalFeatures(keypoints_true[2, 0:12], descriptors2)
        lf1.point_indices = np.array(
            # 0   1  2  3  4   5   6   7   8  9
            [-1, -1, 4, 5, 0, -1, -1, -1, -1, 1]
        )
        lf2.point_indices = np.array(
            # 0   1   2   3  4  5  6  7   8  9  10  11
            [-1, -1, -1, -1, 0, 6, 7, 8, -1, 1, -1, -1]
        )

        with pytest.raises(NotEnoughInliersException):
            get_correspondences(match, [lf1, lf2], lf0)

    case1()
    case2()
    case3()


def test_estimate_pose():
    points = Points()
    points.add(points_true)
    descriptors0 = break_other_than(descriptors,
                                    [2, 3, 4, 5, 6, 7, 9, 12, 13])
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [0, 1, 4, 5, 6, 7, 9, 10, 11, 12, 13])

    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)

    # matches01.T = [[2, 3, 4, 9, 12, 13],
    #                [2, 3, 4, 9, 12, 13]]
    # matches02.T = [[4, 5, 6, 7, 9, 12, 13],
    #                [4, 5, 6, 7, 9, 12, 13]]
    lf1.point_indices = np.array(
        # 0   1  2  3  4   5   6   7   8  9  10  11  12  13
        [-1, -1, 2, 3, 4, -1, -1, -1, -1, 9, -1, -1, 12, 13]
    )
    lf2.point_indices = np.array(
        # 0   1   2   3  4  5  6  7   8  9  10  11  12  13
        [-1, -1, -1, -1, 4, 5, 6, 7, -1, 9, -1, -1, 12, 13]
    )

    point_indices, keypoints = get_correspondences(match, [lf1, lf2], lf0)
    point_indices = [2, 3, 4, 9, 12, 13, 4, 5, 6, 7, 9, 12, 13]
    points_ = points.get(point_indices)
    keypoints0 = keypoints_true[0]
    keypoints = np.vstack([
        keypoints0[2], keypoints0[3], keypoints0[4],
        keypoints0[9], keypoints0[12], keypoints0[13],
        keypoints0[4], keypoints0[5], keypoints0[6],
        keypoints0[7], keypoints0[9], keypoints0[12], keypoints0[13]
    ])

    pose = estimate_pose(match, points, [lf1, lf2], lf0)
    assert_array_almost_equal(pose.R, rotations[0])
    assert_array_almost_equal(pose.t, translations[0])
    assert(pose == Pose(rotations[0], translations[0]))


def assert_projection_equal(points, lf, pose, keypoints):
    point_indices = lf.point_indices[lf.is_triangulated]
    P = transform(pose.R, pose.t, points.get(point_indices))
    assert_array_almost_equal(projection.compute(P),
                              keypoints[lf.is_triangulated])

def test_try_add_more():
    # TODO test with keypoints of different lengths
    vo = VisualOdometry(camera_parameters, FOV(0.0), min_matches=8)

    descriptors0 = break_other_than(descriptors,
                                    [0, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13])
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    descriptors3 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 8, 9])

    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)
    lf3 = LocalFeatures(keypoints_true[3], descriptors3)

    assert(vo.try_add_keyframe(lf0))
    assert(vo.n_active_keyframes == 1)

    assert(vo.try_add_keyframe(lf1))
    assert(vo.n_active_keyframes == 2)

    # first triangulation
    # point_indices   0  1  2  3  4   5   6   7   8
    # matches01.T = [[0, 2, 4, 6, 7, 10, 11, 12, 13],
    #                [0, 2, 4, 6, 7, 10, 11, 12, 13]]

    assert_array_equal(lf0.point_indices,
                       [0, -1, 1, -1, 2, -1, 3, 4, -1, -1, 5, 6, 7, 8])
    assert_array_equal(lf1.point_indices,
                       [0, -1, 1, -1, 2, -1, 3, 4, -1, -1, 5, 6, 7, 8])

    pose0, pose1 = vo.active_poses
    assert_projection_equal(vo.points, lf0, pose0, keypoints_true[0])
    assert_projection_equal(vo.points, lf1, pose1, keypoints_true[1])

    # lf0 and lf1 are already observed and lf2 is added
    # point_indices   2     3  4      5   6   7      existing
    # point_indices      9       10                  newly triangulated
    # matches02.T = [[4, 5, 6, 7, 9, 10, 11, 12],
    #                [4, 5, 6, 7, 9, 10, 11, 12]]
    #
    # point_indices      2  3  4      5   6   7      existing
    # point_indices  11          12                  newly triangulated
    # matches12.T = [[1, 4, 6, 7, 8, 10, 11, 12],
    #                [1, 4, 6, 7, 8, 10, 11, 12]]
    assert(vo.try_add_keyframe(lf2))
    assert(vo.n_active_keyframes == 3)

    lf0, lf1, lf2 = vo.active_local_features
    assert(len(vo.export_points()) == 13)
    assert_array_equal(lf0.point_indices,
                       [0, -1, 1, -1, 2, 9, 3, 4, -1, 10, 5, 6, 7, 8])
    assert_array_equal(lf1.point_indices,
                       [0, 11, 1, -1, 2, -1, 3, 4, 12, -1, 5, 6, 7, 8])
    # [0, 2, 3, 8] cannot be found in matches01 and matches02
    # so they should not be triangulated
    assert_array_equal(lf2.point_indices,
                       [-1, 11, -1, -1, 2, 9, 3, 4, 12, 10, 5, 6, 7, -1])

    pose0, pose1, pose2 = vo.active_poses
    assert_projection_equal(vo.points, lf0, pose0, keypoints_true[0])
    assert_projection_equal(vo.points, lf1, pose1, keypoints_true[1])
    assert_projection_equal(vo.points, lf2, pose2, keypoints_true[2])


def test_try_remove():
    vo = VisualOdometry(camera_parameters, FOV(0.0), min_active_keyframes=4)

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
