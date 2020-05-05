import numpy as np
from numpy.testing import assert_array_equal

from tadataka.camera.parameters import CameraParameters
from tadataka.camera.normalizer import Normalizer


camera_parameters = CameraParameters(focal_length=[10., 20.], offset=[2., 4.])

unnormalized = np.array([
    [12., 24.],
    [0., 0.],
    [8., 10.]
])

normalized = np.array([
    [1.0, 1.0],
    [-0.2, -0.2],
    [0.6, 0.3]
])


def test_normalize():
    assert_array_equal(
        Normalizer(camera_parameters).normalize(unnormalized),
        normalized
    )


def test_unnormalize():
    assert_array_equal(
        Normalizer(camera_parameters).unnormalize(normalized),
        unnormalized
    )
