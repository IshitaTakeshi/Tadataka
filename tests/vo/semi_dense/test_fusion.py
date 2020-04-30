from itertools import product
from numpy.testing import assert_almost_equal
import numpy as np
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.fusion import fusion, fusion_
from tadataka.vo.semi_dense.hypothesis import HypothesisMap


def test_fusion_():
    D1 = np.array([
        [1.9, -2.2],
        [-3.8, 4.1],
        [-1.5, 4.5]
    ])
    D2 = np.array([
        [-4.1, -2.5],
        [1.2, 5.0],
        [6.4, 4.1]
    ])

    V1 = np.array([
        [4.8, 2.2],
        [3.1, 6.8],
        [4.0, 2.1]
    ])
    V2 = np.array([
        [4.2, 3.1],
        [0.01, 2.0],
        [6.0, 3.9]
    ])
    D0, V0 = fusion_(D1, D2, V1, V2)

    for y, x in product(range(0, 3), range(0, 2)):
        d0 = D0[y, x]
        d1 = D1[y, x]
        d2 = D2[y, x]
        var0 = V0[y, x]
        var1 = V1[y, x]
        var2 = V2[y, x]

        assert(d0 == (var2 * d1 + var1 * d2) / (var1 + var2))
        assert(var0 == (var1 * var2) / (var1 + var2))


def test_fusion():
    shape = (10, 10)
    inv_depth_map1 = np.random.uniform(-10, 10, shape)
    inv_depth_map2 = np.random.uniform(-10, 10, shape)
    variance_map1 = np.random.random(shape)
    variance_map2 = np.random.random(shape)

    h1 = HypothesisMap(inv_depth_map1, variance_map1)
    h2 = HypothesisMap(inv_depth_map2, variance_map2)
    h = fusion(h1, h2)

    for (x, y) in image_coordinates(shape):
        inv_depth, variance = fusion_(
            inv_depth_map1[y, x], inv_depth_map2[y, x],
            variance_map1[y, x], variance_map2[y, x]
        )

        assert_almost_equal(h.inv_depth_map[y, x], inv_depth)
        assert_almost_equal(h.variance_map[y, x], variance)
