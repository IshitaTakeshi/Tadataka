import numpy as np
from numpy.testing import assert_array_almost_equal
from tadataka.vo.semi_dense.propagation import (
    new_inverse_depth_map, new_variance_map)


def test_calc_new_inv_depth_map():
    inv_depth_map = np.array([
        [0.25, 0.10],
        [0.20, 0.05]
    ])
    tz = 1.5
    assert_array_almost_equal(
        new_inverse_depth_map(inv_depth_map, tz),
        [[1.0 / (4.0 - 1.5), 1.0 / (10.0 - 1.5)],
         [1.0 / (5.0 - 1.5), 1.0 / (20.0 - 1.5)]]
    )


def test_propaget_inv_depth_map():
    inv_depth_map = np.array([
        [2.5, 1.0],
        [2.0, 0.5]
    ])

    new_inv_depth_map = np.array([
        [5.0, 2.0],
        [3.0, 0.4]
    ])
    variance_map = np.array([
        [6.0, 0.8],
        [1.2, 3.0]
    ])

    V = new_variance_map(inv_depth_map, new_inv_depth_map,
                         variance_map, uncertaintity=1.0)
    assert_array_almost_equal(
        V,
        [[pow(5.0 / 2.5, 4) * 6.0 + 1.0, pow(2.0 / 1.0, 4) * 0.8 + 1.0],
         [pow(3.0 / 2.0, 4) * 1.2 + 1.0, pow(0.4 / 0.5, 4) * 3.0 + 1.0]]
    )
