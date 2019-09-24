from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from vitamine.camera import CameraParameters
from vitamine.projection import pi, PerspectiveProjection


def test_pi():
    P = np.array([
        [1., 2., 2.],
        [2., 1., 4.],
    ])

    expected = np.array([
        [0.5, 1.0],
        [0.5, 0.25]
    ])

    assert_array_equal(pi(P), expected)


def test_perspective_projection():
    camera_parameters = CameraParameters(
        focal_length=[0.8, 1.2],
        offset=[1.0, 2.0]
    )
    projection = PerspectiveProjection(camera_parameters)
    points = np.array([
        [1.0, 2.0, 2.0],
        [4.0, 3.0, 5.0]
    ])

    expected = np.array([
        [1.4, 3.2],
        [1.64, 2.72],
    ])
    assert_array_almost_equal(projection.compute(points), expected)
