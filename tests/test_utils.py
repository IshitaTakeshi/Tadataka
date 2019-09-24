from autograd import numpy as np
from numpy.testing import assert_array_equal
from vitamine.utils import is_in_image_range, radian_to_degree, indices_other_than


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


def test_radian_to_degree():
    assert(np.isclose(radian_to_degree(np.pi / 2), 90.0))
    assert(np.isclose(radian_to_degree(-2 * np.pi / 3), -120.0))


def test_indices_other_than():
    assert_array_equal(indices_other_than(8, [1, 2, 3, 7]), [0, 4, 5, 6])
    assert_array_equal(indices_other_than(5, []), [0, 1, 2, 3, 4])
