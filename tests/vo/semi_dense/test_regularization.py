from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np

from tadataka.vo.semi_dense.hypothesis import HypothesisMap
from tadataka.vo.semi_dense.regularization import (
    regularize, is_statically_same, create_mask_
)


def test_is_statically_same():
    inv_depth_map = np.array([
        [2, -1, 0],
        [-3, 1, 3],
        [4, -2, -4]
    ])
    mask = create_mask_(inv_depth_map, variance=2.0)
    assert_array_equal(
        mask,
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 0]]
    )

    for _ in range(20):
        inv_depth_map = np.random.uniform(-100, 100, (3, 3))
        # center is always 1
        assert(create_mask_(inv_depth_map, 0.01)[1, 1] == 1)


def test_regularization():
    def regularize_(D, V):
        s = 0.0
        d = 0.0
        for y in range(3):
            for x in range(3):
                if not is_statically_same(D[1, 1], D[y, x], V[1, 1], 2.0):
                    continue
                s += D[y, x] / V[y, x]
                d += 1.0 / V[y, x]
        return s / d


    inv_depth_map = np.random.random((4, 4))
    variance_map = np.random.random((4, 4))

    R = regularize(HypothesisMap(inv_depth_map, variance_map), conv_size=3)
    assert(R.shape == (4, 4))
    mask = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ], dtype=np.bool)
    assert_array_equal(R[mask], inv_depth_map[mask])

    assert_almost_equal(
        R[1, 1],
        regularize_(inv_depth_map[0:3, 0:3], variance_map[0:3, 0:3])
    )
    assert_almost_equal(
        R[1, 2],
        regularize_(inv_depth_map[0:3, 1:4], variance_map[0:3, 1:4])
    )
    assert_almost_equal(
        R[2, 1],
        regularize_(inv_depth_map[1:4, 0:3], variance_map[1:4, 0:3])
    )
    assert_almost_equal(
        R[2, 2],
        regularize_(inv_depth_map[1:4, 1:4], variance_map[1:4, 1:4])
    )
