from itertools import product
import numpy as np
from tadataka.vo.semi_dense.fusion import fusion


def test_fusion():
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
    D0, V0 = fusion(D1, D2, V1, V2)

    for y, x in product(range(0, 3), range(0, 2)):
        d0 = D0[y, x]
        d1 = D1[y, x]
        d2 = D2[y, x]
        var0 = V0[y, x]
        var1 = V1[y, x]
        var2 = V2[y, x]

        assert(d0 == (var2 * d1 + var1 * d2) / (var1 + var2))
        assert(var0 == (var1 * var2) / (var1 + var2))
