import numpy as np
from numpy.testing import (assert_array_less, assert_array_equal,
                           assert_array_almost_equal)

from tadataka.camera import CameraParameters
from tadataka.projection import PerspectiveProjection
from tadataka.dataset.observations import generate_translations
from tadataka.rigid_transform import transform_all


points = np.array([
    [1, 2, 3],
    [-1, 0, 1],
    [-2, -1, -1]
])


rotations = np.array([
    [[1, 0, 0],
     [0, -1, 0],
     [0, 0, -1]],
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, -1]]
])


translations = np.array([
    [1, 3, 2],
    [2, 0, -2]
])


assert((np.linalg.det(rotations) == 1).all())


def test_generate_translations():
    def run(offset):
        translations = generate_translations(rotations, points, offset)
        P = transform_all(rotations, translations, points)
        P = P.reshape(-1, 3)
        # check z >= offset for (x, y, z) in P
        # HACK preferable to use 'assert_array_less' instead of 'assert'
        assert((P[:, 2] >= offset).all())

    run(offset=0.0)
    run(offset=2.0)
    run(offset=4.0)
