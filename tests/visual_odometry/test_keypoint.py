from numpy.testing import assert_array_equal
from autograd import numpy as np

from vitamine.visual_odometry.keypoint import (
    LocalFeatures, associate_points, copy_point_indices)
from tests.utils import random_binary


keypoints = np.arange(18).reshape(9, 2)
descriptors = random_binary((9, 256))


def test_is_triangulated():
    lf = LocalFeatures(keypoints, descriptors)
    lf.associate_points([2, 3, 6, 8], [1, 5, 6, 3])
    assert_array_equal(
        lf.is_triangulated,
        np.array([0, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.bool)
    )


def test_triangulated():
    lf = LocalFeatures(keypoints, descriptors)
    lf.associate_points([1, 2, 6, 7], [0, 1, 2, 3])
    keypoints_, descriptors_ = lf.triangulated()

    assert_array_equal(keypoints_, keypoints[[1, 2, 6, 7]])
    assert_array_equal(descriptors_, descriptors[[1, 2, 6, 7]])


def test_untriangulated():
    lf = LocalFeatures(keypoints, descriptors)
    lf.associate_points([1, 2, 6, 7], [0, 1, 2, 3])
    keypoints_, descriptors_ = lf.untriangulated()

    assert_array_equal(keypoints_, keypoints[[0, 3, 4, 5, 8]])
    assert_array_equal(descriptors_, descriptors[[0, 3, 4, 5, 8]])


def test_associate_points():
    matches01 = np.vstack(([2, 4, 8, 1, 6, 7],
                           [1, 2, 4, 5, 6, 8])).T
    point_indices = np.arange(6)

    lf0 = LocalFeatures(keypoints, descriptors)
    lf1 = LocalFeatures(keypoints, descriptors)

    associate_points(lf0, lf1, matches01, point_indices)

    assert_array_equal(lf0.point_indices, [-1, 3, 0, -1, 1, -1, 4, 5, 2])
    assert_array_equal(lf1.point_indices, [-1, 0, 1, -1, 2, 3, 4, -1, 5])


def test_copy_point_indices():
    lf0 = LocalFeatures(keypoints, descriptors)
    lf1 = LocalFeatures(keypoints, descriptors)

    matches01 = np.vstack(([1, 2, 6, 8],
                           [4, 3, 8, 1])).T

    lf0.associate_points([1, 2, 4, 5, 6, 8], np.arange(6))
                                         #  0  1  2   3  4  5  6   7  8
    assert_array_equal(lf0.point_indices, [-1, 0, 1, -1, 2, 3, 4, -1, 5])
    copy_point_indices(lf0, lf1, matches01)
    assert_array_equal(lf1.point_indices, [-1, 5, -1, 1, 0, -1, -1, -1, 4])
