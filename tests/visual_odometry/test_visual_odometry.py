from autograd import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.rigid.transformation import transform
from vitamine.visual_odometry import visual_odometry
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.keypoints import match
from vitamine.exceptions import NotEnoughInliersException
from vitamine.visual_odometry.visual_odometry import (
    VisualOdometry, find_best_match, estimate_pose,
    triangulation, get_correspondences, copy_triangulated)
from vitamine.visual_odometry.pose import Pose
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keypoint import LocalFeatures
from vitamine.rigid.transformation import transform_all
from tests.utils import random_binary


def add_noise(descriptors, indices):
    descriptors = np.copy(descriptors)
    descriptors[indices] = random_binary((len(indices), descriptors.shape[1]))
    return descriptors


def break_other_than(descriptors, indices):
    indices_to_break = np.setxor1d(np.arange(len(descriptors)), indices)
    return add_noise(descriptors, indices_to_break)


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

points_true = np.array([
   [4, -1, 3],   # 0
   [1, -3, 0],   # 1
   [-2, 0, -6],  # 2
   [0, 0, 0],    # 3
   [-3, -2, -5], # 4
   [-3, -1, 8],  # 5
   [-4, -2, 3],  # 6
   [4, 0, 1],    # 7
   [-2, 1, 1],   # 8
   [4, 1, 6],    # 9
   [-4, 4, -1],  # 10
   [-5, 3, 3],   # 11
   [-1, 3, 2],   # 12
   [2, -3, -5]   # 13
])

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

def test_break_other_than():
    descriptors0 = break_other_than(descriptors, [0, 1, 2, 4, 8, 10])
    descriptors1 = add_noise(descriptors, [3, 5, 6, 7, 9, 11, 12, 13])

    assert_array_equal(
        match(descriptors, descriptors0),
        match(descriptors, descriptors1)
    )

    descriptors0 = break_other_than(descriptors, np.arange(4, 14))
    descriptors1 = add_noise(descriptors, np.arange(0, 4))

    assert_array_equal(
        match(descriptors, descriptors0),
        match(descriptors, descriptors1)
    )


def test_find_best_match():
    descriptors_ = [
        descriptors[0:3],  # 3 points can match
        break_other_than(descriptors, [0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13]),
        descriptors[4:8],  # 4 points can match
        break_other_than(descriptors, [0, 1, 2, 3, 4])
    ]

    expected = np.vstack((
        [0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13],
        [0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    )).T
    matches01, argmax = find_best_match(match, descriptors_, descriptors)
    assert_array_equal(matches01, expected)
    assert(argmax == 1)


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

        (R0, t0), (R1, t1) = vo.export_poses()
        points_pred = vo.export_points()

        keypoints_pred0 = projection.compute(transform(R0, t0, points_pred))
        keypoints_pred1 = projection.compute(transform(R1, t1, points_pred))

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
        (R0, t0), (R2, t2) = vo.export_poses()
        keypoints_pred0 = projection.compute(transform(R0, t0, points_pred))
        keypoints_pred2 = projection.compute(transform(R2, t2, points_pred))

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

        point_indices, keypoints = get_correspondences(match, lf0, [lf1, lf2])

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
        keypoints0 = keypoints_true[0]
        lf0 = LocalFeatures(keypoints0, descriptors0)

        point_indices, keypoints = get_correspondences(match, lf0, [lf1, lf2])

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
            get_correspondences(match, lf0, [lf1, lf2])

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

    point_indices, keypoints = get_correspondences(match, lf0, [lf1, lf2])
    point_indices = [2, 3, 4, 9, 12, 13, 4, 5, 6, 7, 9, 12, 13]
    points_ = points.get(point_indices)
    keypoints0 = keypoints_true[0]
    keypoints = np.vstack([
        keypoints0[2], keypoints0[3], keypoints0[4],
        keypoints0[9], keypoints0[12], keypoints0[13],
        keypoints0[4], keypoints0[5], keypoints0[6],
        keypoints0[7], keypoints0[9], keypoints0[12], keypoints0[13]
    ])

    from vitamine import pose_estimation as PE
    pose = estimate_pose(match, points, lf0, [lf1, lf2])
    assert_array_almost_equal(pose.R, rotations[0])
    assert_array_almost_equal(pose.t, translations[0])
    assert(pose == Pose(rotations[0], translations[0]))


def test_try_add_more():
    # TODO test with keypoints of different lengths
    vo = VisualOdometry(camera_parameters, FOV(0.0), min_matches=8)

    descriptors0 = break_other_than(descriptors,
                                    [0, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13])
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)

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


def test_copy_triangulated():
    descriptors0 = break_other_than(descriptors,
                                    [0, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13])
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 13])
    descriptors2 = break_other_than(descriptors,
                                    [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # point_indices   0  1  2  3  4   5   6   7   8
    # matches01.T = [[0, 2, 4, 6, 7, 10, 11, 12, 13],
    #                [0, 2, 4, 6, 7, 10, 11, 12, 13]]
    #
    # point_indices   2  8  3  4  9   5   6   7
    # matches02.T = [[4, 5, 6, 7, 9, 10, 11, 12],
    #                [4, 5, 6, 7, 9, 10, 11, 12]]
    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)
    lf1.point_indices = np.array(
        [0, -1, 1, -1, 2, -1, 3, 4, -1, -1, 5, 6, 7, 8]
    )
    lf2.point_indices = np.array(
        [-1, -1, -1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, -1]
    )

    copy_triangulated(match, [lf1, lf2], lf0)
    assert_array_equal(lf0.point_indices,
                       [0, -1, 1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, 8])


def test_triangulation():
    # test the case that lf1, lf2, lf3 are already observed and lf0 is added
    # as a new keyframe
    pose0 = Pose(rotations[0], translations[0])
    pose1 = Pose(rotations[1], translations[1])
    pose2 = Pose(rotations[2], translations[2])
    pose3 = Pose(rotations[3], translations[3])

    descriptors0 = break_other_than(descriptors,
                                    [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13])
    # point_indices   0  1  2  3  4   5   6
    # matches01.T = [[0, 1, 4, 7, 9, 12, 13],
    #                [0, 1, 4, 7, 9, 12, 13]]
    descriptors1 = break_other_than(descriptors,
                                    [0, 1, 2, 3, 4, 7, 9, 12, 13])
    # point_indices   0  1  2     3  4       5   6   existing
    # point_indices            7         8           newly triangulated
    # matches02.T = [[0, 1, 4, 5, 7, 9, 11, 12, 13],
    #                [0, 1, 4, 5, 7, 9, 11, 12, 13]]
    descriptors2 = break_other_than(descriptors,
                                    [0, 1, 4, 5, 7, 9, 11, 12, 13])
    # point_indices   0  1  7  3  4       6   existing
    # point_indices                   9       newly triangulated
    # matches03.T = [[0, 1, 5, 7, 9, 10, 13],
    #                [0, 1, 5, 7, 9, 10, 13]]
    descriptors3 = break_other_than(descriptors,
                                    [0, 1, 2, 5, 7, 9, 10, 13])

    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)
    lf3 = LocalFeatures(keypoints_true[3], descriptors3)
    points = Points()
    triangulation(match, points,
                  [pose1, pose2, pose3], [lf1, lf2, lf3], pose0, lf0)
    assert_array_equal(lf0.point_indices,
                       [0, 1, -1, -1, 2, 7, -1, 3, -1, 4, 9, 8, 5, 6])
    assert_array_equal(lf1.point_indices,
                       [0, 1, -1, -1, 2, -1, -1, 3, -1, 4, -1, -1, 5, 6])
    assert_array_equal(lf2.point_indices,
                       [0, 1, -1, -1, 2, 7, -1, 3, -1, 4, -1, 8, 5, 6])
    assert_array_equal(lf3.point_indices,
                       [0, 1, -1, -1, -1, 7, -1, 3, -1, 4, 9, -1, -1, 6])
