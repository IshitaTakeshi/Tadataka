from numpy.testing import assert_almost_equal
import numpy as np

from tadataka.vo.semi_dense.regularization import regularize


def test_regularization():
    def regularize_(inv_depth_map_, variance_map_):
        s = 0.0
        d = 0.0
        for y in range(3):
            for x in range(3):
                s += inv_depth_map_[y, x] / variance_map_[y, x]
                d += 1.0 / variance_map_[y, x]
        return s / d


    inv_depth_map = np.array([
        [4, 2, 4, 3],
        [2, 1, 5, 2],
        [3, 4, 1, 2],
        [4, 1, 2, 3]
    ])
    variance_map = np.array([
        [3, 2, 2, 1],
        [4, 9, 3, 5],
        [3, 2, 8, 2],
        [7, 5, 3, 7]
    ])

    R = regularize(inv_depth_map, variance_map)

    assert_almost_equal(
        R[0, 0],
        regularize_(inv_depth_map[0:3, 0:3], variance_map[0:3, 0:3])
    )
    assert_almost_equal(
        R[0, 1],
        regularize_(inv_depth_map[0:3, 1:4], variance_map[0:3, 1:4])
    )
    assert_almost_equal(
        R[1, 0],
        regularize_(inv_depth_map[1:4, 0:3], variance_map[1:4, 0:3])
    )
    assert_almost_equal(
        R[1, 1],
        regularize_(inv_depth_map[1:4, 1:4], variance_map[1:4, 1:4])
    )
