from numpy.testing import assert_array_almost_equal
from autograd import numpy as np
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import Normalizer, FOV, calc_factors


def test_normalizer():
    camera_parameters = CameraParameters(
        focal_length=[2., 3.],
        offset=[-1., 4.]
    )
    keypoints = np.array([
        [4, 3],
        [2, 1],
        [1, 9]
    ])
    expected = np.array([
        [5 / 2, -1 / 3],
        [3 / 2, -3 / 3],
        [2 / 2, 5 / 3]
    ])
    assert_array_almost_equal(
        Normalizer(camera_parameters).normalize(keypoints),
        expected
    )


def test_fov_undistort():
    X = np.array([
        [0.0, 0.0],
        [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],  # r = 1
        [1.0, np.sqrt(3)]  # r = 2
    ])

    expected = np.array([
        # (pi / 3) / (2 * tan(pi / 6))
        (np.pi / 3) / (2 * np.sqrt(3) / 3),
        # tan(1 * pi / 3) / (2 * 1 * tan(pi / 6))
        np.sqrt(3) / (2 * np.sqrt(3) / 3),
        # tan(2 * pi / 3) / (2 * 2 * tan(pi / 6))
        -np.sqrt(3) / (2 * 2 * np.sqrt(3) / 3)
    ])

    # test 'calc_factors' separately because
    # we cannot test factors for X[0] using FOV.undistort
    assert_array_almost_equal(
        calc_factors(X, omega=np.pi/3),
        expected
    )
    assert_array_almost_equal(
        FOV(omega=np.pi/3).undistort(X),
        expected.reshape(-1, 1) * X
    )

    Y = FOV(omega=0).undistort(X)
    # all factors = 1
    assert_array_almost_equal(Y, X)
