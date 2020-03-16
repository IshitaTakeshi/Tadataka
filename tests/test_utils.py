import numpy as np
from numpy.testing import assert_array_equal
from tadataka.utils import (
    merge_dicts, is_in_image_range, radian_to_degree, indices_other_than)
from tadataka.utils import add_noise, break_other_than, random_binary


def test_merge_dicts():
    d1 = {'a': 1, 'b': 2}
    d2 = {'c': 2, 'd': 3}
    d3 = {'e': 3, 'f': 4, 'g': 5}
    d = merge_dicts(d1, d2, d3)
    expected = {'a': 1, 'b': 2, 'c': 2, 'd': 3, 'e': 3, 'f': 4, 'g': 5}
    assert(d == expected)


def test_is_in_image_range():
    width, height = 20, 30
    image_shape = (height, width)

    keypoints = np.array([
    #     x   y
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
        is_in_image_range(keypoints, image_shape),
        expected
    )

    # case if keypoints are in float
    keypoints = np.array([
        #    x      y
        [19.00, 29.00],
        [19.01, 29.00],
        [19.00, 29.01],
        [19.01, 29.01],
        [00.00, 00.00],
        [00.00, -0.01],
        [-0.01, 00.00],
        [-0.01, -0.01],
    ])

    expected = np.array([True, False, False, False, True, False, False, False])

    assert_array_equal(
        is_in_image_range(keypoints, image_shape),
        expected
    )

    # case if 1d array is passed
    assert(is_in_image_range([0, 29], image_shape))
    assert(not is_in_image_range([-1, 29], image_shape))


def test_radian_to_degree():
    assert(np.isclose(radian_to_degree(np.pi / 2), 90.0))
    assert(np.isclose(radian_to_degree(-2 * np.pi / 3), -120.0))


def test_indices_other_than():
    assert_array_equal(indices_other_than(8, [1, 2, 3, 7]), [0, 4, 5, 6])
    assert_array_equal(indices_other_than(5, []), [0, 1, 2, 3, 4])
