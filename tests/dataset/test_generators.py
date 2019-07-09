import numpy as np
from numpy.testing import assert_array_less, assert_array_equal

from camera import CameraParameters
from projection.projections import PerspectiveProjection
from dataset.generators import generate_translations, generate_observations
from rigid.transformation import transform_each


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
        P = transform_each(rotations, translations, points)
        P = P.reshape(-1, 3)
        # check z >= offset for (x, y, z) in P
        # HACK preferable to use 'assert_array_less' instead of 'assert'
        assert((P[:, 2] >= offset).all())

    run(offset=0.0)
    run(offset=2.0)
    run(offset=4.0)


def test_generate_observations():
    camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
    projection = PerspectiveProjection(camera_parameters)
    observations = generate_observations(rotations, translations, points,
                                         projection)
    expected = np.array([
        [[2 / -1, 1 / -1],
         [0 / 1, 3 / 1],
         [-1 / 3, 4 / 3]],
        [[4 / -5, 1 / -5],
         [2 / -3, -1 / -3],
         [1 / -1, -2 / -1]],
    ])

    assert_array_equal(observations, expected)
