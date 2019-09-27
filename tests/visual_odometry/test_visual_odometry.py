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
from vitamine.visual_odometry.visual_odometry import (
    VisualOdometry, find_best_match,
    triangulation, match_triangulate)
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



def test_initialize_second():
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
    case1()
    case2()


def test_try_add_more():
    return
    # TODO test with keypoints of different lengths
    vo = VisualOdometry(camera_parameters, FOV(0.0), min_matches=8)

    descriptors0 = add_noise(descriptors, [0, 1])
    descriptors1 = add_noise(descriptors, [5])
    descriptors2 = add_noise(descriptors, [3])
    lf0 = LocalFeatures(keypoints_true[0], descriptors0)
    lf1 = LocalFeatures(keypoints_true[1], descriptors1)
    lf2 = LocalFeatures(keypoints_true[2], descriptors2)

    vo.try_add_keyframe(lf0)
    assert(vo.n_active_keyframes == 1)
    assert(vo.try_add_keyframe(lf1))
    assert(vo.n_active_keyframes == 2)

    assert_array_equal(lf0.point_indices,
                       [-1, -1, 0, 1, 2, -1, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_array_equal(lf1.point_indices,
                       [-1, -1, 0, 1, 2, -1, 3, 4, 5, 6, 7, 8, 9, 10])

    assert(vo.try_add_keyframe(lf2))
    assert(vo.n_active_keyframes == 3)

    lf0, lf1, lf2 = vo.local_features
    assert_array_equal(lf0.point_indices,
                       [11, 12, 0, 1, 2, 13, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_array_equal(lf1.point_indices,
                       [11, 12, 0, 1, 2, 13, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_array_equal(lf2.point_indices,
                       [11, 12, 0, -1, 2, 13, 3, 4, 5, 6, 7, 8, 9, 10])


def test_triangulation():
    pass
