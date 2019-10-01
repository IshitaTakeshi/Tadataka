from numpy.testing import assert_array_almost_equal
from autograd import numpy as np
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import Normalizer, FOV, calc_factors
from vitamine.projection import PerspectiveProjection
from vitamine.rigid_transform import transform
from vitamine.pose_estimation import solve_pnp
from vitamine.so3 import rodrigues
from tests.data import dummy_points as points_true


def test_normalizer():
    camera_parameters = CameraParameters(
        focal_length=[2., 3.],
        offset=[-1., 4.]
    )
    projection = PerspectiveProjection(camera_parameters)

    normalizer = Normalizer(camera_parameters)

    def case1():
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
            normalizer.normalize(keypoints),
            expected
        )

    def case2():
        omega_true = np.array([-1.0, 0.2, 3.1])
        t_true = np.array([-0.8, 1.0, 8.3])

        R = rodrigues(np.atleast_2d(omega_true))[0]
        P = transform(R, t_true, points_true)
        keypoints_true = projection.compute(P)

        # poses should be able to be estimated without a camera matrix
        keypoints_ = normalizer.normalize(keypoints_true)
        omega_pred, t_pred = solve_pnp(points_true, keypoints_,
                                       omega_true, t_true)

        assert_array_almost_equal(omega_true, omega_pred)
        assert_array_almost_equal(t_true, t_pred)


    case1()
    case2()


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
