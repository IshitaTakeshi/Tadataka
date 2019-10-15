from autograd import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from vitamine.camera import CameraParameters
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.keypoints import Matcher
from vitamine.projection import PerspectiveProjection
from vitamine.visual_odometry.triangulation import (
    triangulation, copy_triangulated)
from vitamine.pose import Pose
from vitamine.rigid_transform import transform
from vitamine.so3 import rodrigues
from vitamine.utils import random_binary, break_other_than
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keypoint import is_triangulated, init_point_indices
from tests.data import dummy_points as points_true


matcher = Matcher(enable_ransac=False, enable_homography_filter=False)

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


descriptors = random_binary((14, 1024))


def test_copy_triangulated():
    def case1():
        # point_indices        0  1  2  3  4   5   6   7   8
        matches01 = np.array([[0, 2, 4, 6, 7, 10, 11, 12, 13],
                              [0, 2, 4, 6, 7, 10, 11, 12, 13]]).T

        # point_indices        2  8  3  4  9   5   6   7
        matches02 = np.array([[4, 5, 6, 7, 9, 10, 11, 12],
                              [4, 5, 6, 7, 9, 10, 11, 12]]).T

        point_indices0 = init_point_indices(14)
        point_indices1 = np.array(
        #    0   1  2   3  4   5  6  7   8  9  10 11 12 13
            [0, -1, 1, -1, 2, -1, 3, 4, -1, -1, 5, 6, 7, 8]
        )
        point_indices2 = np.array(
        #     0   1   2   3  4  5  6  7   8  9 10 11 12  13
            [-1, -1, -1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, -1]
        )

        copy_triangulated([matches01, matches02],
                          [point_indices1, point_indices2],
                          point_indices0)
        assert_array_equal(
            point_indices0,
        #    0   1  2   3  4  5  6  7   8  9 10 11 12 13
            [0, -1, 1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, 8]
        )

    def case2():
        # point_indices        0  1  2   5   6   7   8
        matches01 = np.array([[0, 2, 4, 10, 11, 12, 13],
                              [0, 2, 4, 10, 11, 12, 13]]).T
        # point_indices        2  8  3  4  9   5   6   7
        matches02 = np.array([[4, 5, 6, 7, 9, 10, 11, 12],
                              [4, 5, 6, 7, 9, 10, 11, 12]]).T
        point_indices0 = init_point_indices(14)
        point_indices1 = np.array(
        #    0   1  2   3  4   5   6   7   8   9 10 11 12 13
            [0, -1, 1, -1, 2, -1, -1, -1, -1, -1, 5, 6, 7, 8]
        )
        point_indices2 = np.array(
        #     0   1   2   3  4  5  6  7   8  9 10 11 12  13
            [-1, -1, -1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, -1]
        )

        copy_triangulated([matches01, matches02],
                          [point_indices1, point_indices2],
                          point_indices0)
        assert_array_equal(
            point_indices0,
        #    0   1  2   3  4  5  6  7   8  9 10 11 12 13
            [0, -1, 1, -1, 2, 8, 3, 4, -1, 9, 5, 6, 7, 8]
        )

    case1()
    case2()


def test_triangulation():
    pose0 = Pose(omegas[0], translations[0])
    pose1 = Pose(omegas[1], translations[1])
    pose2 = Pose(omegas[2], translations[2])
    pose3 = Pose(omegas[3], translations[3])
    pose4 = Pose(omegas[4], translations[4])

    def case1():
        # point_indices        0  1  2  3  4   5   6   newly triangulated
        matches01 = np.array([[0, 1, 4, 7, 9, 12, 13],
                              [0, 1, 4, 7, 9, 12, 13]]).T

        # point_indices        0  1  2     3  4       5   6   existing
        # point_indices                 7         8           newly triangulated
        matches02 = np.array([[0, 1, 4, 5, 7, 9, 11, 12, 13],
                              [0, 1, 4, 5, 7, 9, 11, 12, 13]]).T
        # point_indices        0  1  7  3  4       6   existing
        # point_indices                        9       newly triangulated
        matches03 = np.array([[0, 1, 5, 7, 9, 10, 13],
                              [0, 1, 5, 7, 9, 10, 13]]).T

        # point_indices        0  1  2  3  4   9   6   existing
        matches04 = np.array([[0, 1, 4, 7, 9, 10, 13],
                              [0, 1, 4, 7, 9, 10, 13]]).T

        keypoints0 = keypoints_true[0]
        keypoints1 = keypoints_true[1]
        keypoints2 = keypoints_true[2]
        keypoints3 = keypoints_true[3]
        keypoints4 = keypoints_true[4]

        point_indices0 = init_point_indices(14)
        point_indices1 = init_point_indices(14)
        point_indices2 = init_point_indices(14)
        point_indices3 = init_point_indices(14)
        point_indices4 = init_point_indices(14)

        points = Points()
        triangulation(
            points,
            [matches01, matches02, matches03, matches04],
            [pose1, pose2, pose3, pose4],
            [keypoints1, keypoints2, keypoints3, keypoints4],
            [point_indices1, point_indices2, point_indices3, point_indices4],
            pose0, keypoints0, point_indices0
        )

        assert_array_equal(point_indices0,
        #                   0  1   2   3  4  5   6  7   8  9 10 11 12 13
                           [0, 1, -1, -1, 2, 7, -1, 3, -1, 4, 9, 8, 5, 6])
        assert_array_equal(point_indices1,
        #                   0  1   2   3  4   5   6  7   8  9  10  11 12 13
                           [0, 1, -1, -1, 2, -1, -1, 3, -1, 4, -1, -1, 5, 6])
        assert_array_equal(point_indices2,
        #                   0  1   2   3  4  5   6  7   8  9  10 11 12 13
                           [0, 1, -1, -1, 2, 7, -1, 3, -1, 4, -1, 8, 5, 6])
        assert_array_equal(point_indices3,
        #                   0  1   2   3   4  5   6  7   8  9 10  11  12 13
                           [0, 1, -1, -1, -1, 7, -1, 3, -1, 4, 9, -1, -1, 6])
        assert_array_equal(point_indices4,
        #                   0  1   2   3  4   5   6  7   8  9 10  11  12 13
                           [0, 1, -1, -1, 2, -1, -1, 3, -1, 4, 9, -1, -1, 6])

        mask0 = is_triangulated(point_indices0)
        mask1 = is_triangulated(point_indices1)
        mask2 = is_triangulated(point_indices2)
        mask3 = is_triangulated(point_indices3)

        P0 = transform(pose0.R, pose0.t, points.get(point_indices0[mask0]))
        P1 = transform(pose1.R, pose1.t, points.get(point_indices1[mask1]))
        P2 = transform(pose2.R, pose2.t, points.get(point_indices2[mask2]))
        P3 = transform(pose3.R, pose3.t, points.get(point_indices3[mask3]))
        assert_array_almost_equal(projection.compute(P0), keypoints0[mask0])
        assert_array_almost_equal(projection.compute(P1), keypoints1[mask1])
        assert_array_almost_equal(projection.compute(P2), keypoints2[mask2])
        assert_array_almost_equal(projection.compute(P3), keypoints3[mask3])

    def case2():
        # point_indices   0  1  2  3  4
        matches01 = np.array([[0, 1, 4, 7, 9],
                              [0, 1, 4, 7, 9]]).T
        # point_indices   0  1  2     3  4   existing
        # point_indices            5         newly triangulated
        matches02 = np.array([[0, 1, 4, 5, 7, 9],
                              [0, 1, 4, 5, 7, 9]]).T

        keypoints0 = keypoints_true[0, 0:10]
        keypoints1 = keypoints_true[1, 0:14]
        keypoints2 = keypoints_true[2, 0:12]
        points = Points()

        point_indices0 = init_point_indices(10)
        point_indices1 = init_point_indices(14)
        point_indices2 = init_point_indices(12)

        triangulation(
            points,
            [matches01, matches02], [pose1, pose2],
            [keypoints1, keypoints2], [point_indices1, point_indices2],
            pose0, keypoints0, point_indices0
        )
        assert_array_equal(point_indices0,
                           [0, 1, -1, -1, 2, 5, -1, 3, -1, 4])
        assert_array_equal(point_indices1,
                           [0, 1, -1, -1, 2, -1, -1, 3, -1, 4, -1, -1, -1, -1])
        assert_array_equal(point_indices2,
                           [0, 1, -1, -1, 2, 5, -1, 3, -1, 4, -1, -1])

        mask0 = is_triangulated(point_indices0)
        mask1 = is_triangulated(point_indices1)
        mask2 = is_triangulated(point_indices2)

        P0 = transform(pose0.R, pose0.t, points.get(point_indices0[mask0]))
        P1 = transform(pose1.R, pose1.t, points.get(point_indices1[mask1]))
        P2 = transform(pose2.R, pose2.t, points.get(point_indices2[mask2]))
        assert_array_almost_equal(projection.compute(P0), keypoints0[mask0])
        assert_array_almost_equal(projection.compute(P1), keypoints1[mask1])
        assert_array_almost_equal(projection.compute(P2), keypoints2[mask2])

    case1()
    case2()
