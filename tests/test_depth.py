from numpy.testing import assert_array_equal
import numpy as np
from tadataka.depth import compute_depth_mask


def test_compute_depth_mask():
    depths = np.array([
        [-1, 4, 2, 3, -4],
        [-8, 5, 1, 0, 2]
    ])

    assert_array_equal(
        compute_depth_mask(depths, min_depth=0.0),
        [False, True, True, False, False]
    )

    assert_array_equal(
        compute_depth_mask(depths, min_depth=1.0),
        [False, True, False, False, False]
    )
