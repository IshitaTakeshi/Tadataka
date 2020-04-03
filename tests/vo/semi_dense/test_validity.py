from numpy.testing import assert_array_almost_equal
import numpy as np
from tadataka.vo.semi_dense import validity


def test_decrease():
    validity_map = np.array([
        [0.4, 0.9, 1.0],
        [0.8, 0.0, 0.1]
    ])
    mask = np.array([
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    assert_array_almost_equal(
        validity.decrease(validity_map, mask, rate=0.8),
        [[0.32, 0.90, 0.80],
         [0.80, 0.00, 0.08]]
    )


def test_increase():
    validity_map = np.array([
        [0.4, 0.9, 1.0],
        [0.8, 0.0, 0.1]
    ])
    mask = np.array([
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    assert_array_almost_equal(
        validity.increase(validity_map, mask, rate=1.2),
        [[0.48, 0.90, 1.00],
         [0.80, 0.00, 0.12]]
    )
