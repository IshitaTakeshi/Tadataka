import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from tadataka.vo.semi_dense.depth import (
    calc_ref_inv_depth, calc_key_depth, depth_search_range,
    InvDepthSearchRange)
from tadataka.vo.semi_dense.hypothesis import Hypothesis


def test_calc_ref_inv_depth():
    depth_key = 4.0
    x_key = np.array([0.5, 2.0])
    T_rk = np.array([
        [0, 0, 1, 3],
        [0, 1, 0, 2],
        [-1, 0, 0, 4],
        [0, 0, 0, 1]
    ])
    assert_almost_equal(calc_ref_inv_depth(T_rk, x_key, 1 / depth_key),
                        1 / (-2.0 + 4.0))


def test_inv_depth_search_range():
    # predicted range fits in [min, max]
    search_range = InvDepthSearchRange(0.4, 1.0, factor=2.0)
    assert_array_almost_equal(search_range(Hypothesis(0.7, 0.1)), (0.5, 0.9))

    # inv_depth - factor * variance < min < inv_depth + factor * variance
    assert_array_almost_equal(search_range(Hypothesis(0.3, 0.1)), (0.4, 0.5))
    # inv_depth - factor * variance < max < inv_depth + factor * variance
    assert_array_almost_equal(search_range(Hypothesis(0.9, 0.1)), (0.7, 1.0))

    # inv_depth - factor * variance < inv_depth + factor * variance < min
    assert(search_range(Hypothesis(0.0, 0.1)) is None)
    # max < inv_depth - factor * variance < inv_depth + factor * variance
    assert(search_range(Hypothesis(1.5, 0.1)) is None)

    # border test
    # inv_depth - factor * variance < inv_depth + factor * variance <= min
    assert(search_range(Hypothesis(0.2, 0.1)) is None)
    # max <= inv_depth - factor * variance < inv_depth + factor * variance
    assert(search_range(Hypothesis(1.2, 0.1)) is None)


def test_depth_search_range():
    min_depth, max_depth = depth_search_range(0.05, 200.0)
    assert_almost_equal(min_depth, 0.005)
    assert_almost_equal(max_depth, 20.0)
