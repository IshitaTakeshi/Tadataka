from autograd import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from vitamine.camera import CameraParameters
from vitamine.projection import PerspectiveProjection
from vitamine.point_index import PointIndices
from vitamine.points import Points
from vitamine.visual_odometry.pose import get_correspondences, estimate_pose
from vitamine.utils import random_binary
from vitamine.rigid_transform import transform
from vitamine.so3 import exp_so3
from tests.data import dummy_points as points_true


def test_get_correspondences():
    keypoints_true = np.arange(2 * 100 * 3).reshape(2, 100, 3)

    point_indices1 = PointIndices(14)
    point_indices2 = PointIndices(14)
    point_indices1.set_triangulated(
        np.array([2, 3, 4, 9, 12, 13]),
        np.array([4, 5, 0, 1, 2, 3])
    )
    point_indices2.set_triangulated(
        np.array([4, 5, 6, 7, 9, 12, 13]),
        np.array([0, 6, 7, 8, 1, 2, 3])
    )

    def case1():
        #                      4  5  0  1   2   3
        matches01 = np.array([[1, 3, 8, 9, 12, 13],
                              [2, 3, 4, 9, 12, 13]]).T
        #                      0  6  7  8  1   2   3
        matches02 = np.array([[4, 5, 6, 7, 9, 12, 13],
                              [4, 5, 6, 7, 9, 12, 13]]).T
        matches = np.vstack((matches01, matches02))

        keypoints0 = keypoints_true[0]

        point_indices, keypoint_indices = get_correspondences(
            [matches01, matches02],
            [point_indices1, point_indices2]
        )

        expected = np.array([1, 3, 8, 9, 12, 13, 4, 5, 6, 7, 9, 12, 13])
        assert_array_equal(keypoint_indices, expected)

        expected = np.array([4, 5, 0, 1, 2, 3, 0, 6, 7, 8, 1, 2, 3])
        assert_array_equal(point_indices, expected)

    def case2():
        #                      4  5  0  1   2   3
        matches01 = np.array([[2, 3, 4, 9, 12, 10],
                              [2, 3, 4, 9, 12, 13]]).T
        #                      0  6  7  8  1   2   3
        matches02 = np.array([[4, 5, 6, 7, 9, 12, 13],
                              [4, 5, 6, 7, 9, 12, 13]]).T

        keypoints0 = keypoints_true[0]

        point_indices, keypoint_indices = get_correspondences(
            [matches01, matches02],
            [point_indices1, point_indices2]
        )

        expected = np.array([2, 3, 4, 9, 12, 10, 4, 5, 6, 7, 9, 12, 13])
        assert_array_equal(keypoint_indices, expected)

        expected = np.array([4, 5, 0, 1, 2, 3, 0, 6, 7, 8, 1, 2, 3])
        assert_array_equal(point_indices, expected)

    case1()
    case2()


def test_estimate_pose():
    camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
    projection = PerspectiveProjection(camera_parameters)

    points = Points()
    points.add(points_true)

    matches01 = np.array([[2, 3, 4, 9, 12, 13],
                          [2, 3, 4, 8, 12, 11]]).T
    matches02 = np.array([[4, 5, 6, 7, 9, 12, 13],
                          [4, 5, 6, 8, 9, 10, 13]]).T

    point_indices1 = PointIndices(14)
    point_indices2 = PointIndices(14)

    point_indices1.set_triangulated(
        np.array([2, 3, 4, 8, 11, 12]),
        np.array([2, 3, 4, 9, 13, 12])
    )
    point_indices2.set_triangulated(
        np.array([4, 5, 6, 8, 9, 10, 13]),
        np.array([4, 5, 6, 7, 9, 12, 13])
    )
    omega = np.array([0, -np.pi / 8, np.pi / 2])
    t = np.array([-5, 3, 8])

    keypoints0 = projection.compute(transform(exp_so3(omega), t, points_true))

    pose = estimate_pose(points, [matches01, matches02],
                         [point_indices1, point_indices2], keypoints0)

    assert_array_almost_equal(pose.omega, omega)
    assert_array_almost_equal(pose.t, t)
