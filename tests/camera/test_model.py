import numpy as np
from numpy.testing import assert_array_equal
from tadataka.camera import CameraModel, CameraParameters


def test_camera_model():
    camera_model = CameraModel(
        CameraParameters(focal_length=[4, 3], offset=[20, 18]),
        distortion_model=None
    )

    assert_array_equal(
        camera_model.normalize(np.array([28, 24])),
        [(28 - 20) / 4, (24 - 18) / 3]
    )

    assert_array_equal(
        camera_model.unnormalize(np.array([2.5, 4.2])),
        [2.5 * 4 + 20, 4.2 * 3 + 18]
    )
