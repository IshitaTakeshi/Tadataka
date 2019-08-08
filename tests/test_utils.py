from autograd import numpy as np
from numpy.testing import assert_array_equal
from vitamine.util import is_in_image_range


def test_is_in_image_range():
    height, width = 30, 20

    keypoints = np.array([
        [19, 29],
        [19, 0],
        [0, 29],
        [-1, 29],
        [19, -1],
        [20, 29],
        [19, 30],
        [20, 30]
    ])

    expected = np.array([True, True, True, False, False, False, False, False])

    assert_array_equal(
        is_in_image_range(keypoints, (height, width)),
        expected
    )
