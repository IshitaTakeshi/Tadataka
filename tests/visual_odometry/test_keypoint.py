from numpy.testing import assert_array_equal
from autograd import numpy as np

from vitamine.visual_odometry.keypoint import KeypointManager, PointIndices
from tests.utils import random_binary


keypoints = np.arange(16).reshape(8, 2)
descriptors = random_binary((keypoints.shape[0], 128))


def test_point_indices():
    p = PointIndices(9)
    p.subscribe([2, 3, 6, 8], [1, 5, 6, 3])
    assert_array_equal(
        p.is_triangulated,
        np.array([0, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.bool)
    )


def test_keypoint_manager():
    manager = KeypointManager()

    manager.add(keypoints, descriptors)
    manager.add_triangulated(0, [1, 2, 6, 7], [0, 1, 2, 3])
    keypoints_, descriptors_ = manager.get_triangulated(0)

    assert_array_equal(keypoints_, keypoints[[1, 2, 6, 7]])
    assert_array_equal(descriptors_, descriptors[[1, 2, 6, 7]])
    assert_array_equal(manager.get_point_indices(0), [0, 1, 2, 3])
