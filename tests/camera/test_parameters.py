import numpy as np
from numpy.testing import assert_array_equal
from tadataka.camera import CameraParameters


def test_camera():
    camera_parameters = CameraParameters(
        image_shape=[100, 100],  # temporal values
        focal_length=[1.0, 1.2],
        offset=[0.8, 0.2]
    )

    expected = np.array([
        [1.0, 0.0, 0.8],
        [0.0, 1.2, 0.2],
        [0.0, 0.0, 1.0]
    ])

    assert_array_equal(camera_parameters.matrix, expected)
