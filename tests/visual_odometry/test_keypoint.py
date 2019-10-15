from numpy.testing import assert_array_equal
from autograd import numpy as np

from vitamine.visual_odometry.keypoint import LocalFeatures
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
