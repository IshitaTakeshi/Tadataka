import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tadataka.camera import CameraModel, CameraParameters, FOV, resize


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


def test_resize():
    distortion_model = FOV(0.02)
    camera_model = CameraModel(
        CameraParameters(focal_length=[40, 48], offset=[20, 16]),
        distortion_model=distortion_model
    )

    resized = resize(camera_model, 1/4)
    assert_array_almost_equal(resized.camera_parameters.focal_length, [10, 12])
    assert_array_almost_equal(resized.camera_parameters.offset, [5, 4])
    assert(resized.distortion_model is distortion_model)
