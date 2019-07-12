import numpy as np
from numpy.testing import assert_array_equal

from vitamine.bundle_adjustment.mask import (
    mask, keypoint_mask, point_mask, pose_mask
)


def test_mask():
    array = np.array([
        [1, 1],
        [np.nan, 1],
        [1, np.nan],
        [np.nan, np.nan],
    ])

    expected = np.array([True, False, False, False])
    assert_array_equal(mask(array), expected)


def test_keypoint_mask():
    array = np.array([
        [[1, 1],
         [np.nan, 1],
         [1, np.nan],
         [np.nan, np.nan]],
        [[1, 1],
         [np.nan, 1],
         [1, np.nan],
         [np.nan, np.nan]]
    ])

    expected = np.array([
        [True, False, False, False],
        [True, False, False, False]
    ])
    assert_array_equal(keypoint_mask(array), expected)


def test_point_mask():
    array = np.array([
        [1, 1, 1],
        [1, 1, np.nan],
        [1, np.nan, 1],
        [np.nan, np.nan, np.nan]
    ])

    expected = np.array([True, False, False, False])
    assert_array_equal(point_mask(array), expected)


def test_pose_mask():
    omegas = np.array([
        [1, 1, 1],
        [1, 1, np.nan],
        [1, 1, 1],
        [1, 1, np.nan]
    ])

    translations = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, np.nan],
        [1, 1, np.nan]
    ])

    expected = np.array([True, False, False, False])
    assert_array_equal(pose_mask(omegas, translations), expected)
