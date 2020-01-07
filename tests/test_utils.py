import numpy as np
from numpy.testing import assert_array_equal
from tadataka.utils import (
    merge_dicts, is_in_image_range, radian_to_degree, indices_other_than)
from tadataka.feature import match
from tadataka.utils import add_noise, break_other_than, random_binary


def test_break_other_than():
    descriptors = random_binary((14, 1024))
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


def test_merge_dicts():
    d1 = {'a': 1, 'b': 2}
    d2 = {'c': 2, 'd': 3}
    d3 = {'e': 3, 'f': 4, 'g': 5}
    d = merge_dicts(d1, d2, d3)
    expected = {'a': 1, 'b': 2, 'c': 2, 'd': 3, 'e': 3, 'f': 4, 'g': 5}
    assert(d == expected)


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
