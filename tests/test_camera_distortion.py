from numpy.testing import assert_array_almost_equal
from autograd import numpy as np
from tadataka.camera import CameraParameters
from tadataka.camera_distortion import (
    Normalizer, FOV, distort_factors, undistort_factors)
from tadataka.projection import PerspectiveProjection
from tadataka.rigid_transform import transform
from tadataka.pose import solve_pnp
from tadataka.so3 import exp_so3
from tests.data import dummy_points as points


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

        P = transform(exp_so3(omega_true), t_true, points)
        keypoints_true = projection.compute(P)

        # poses should be able to be estimated without a camera matrix
        keypoints_ = normalizer.normalize(keypoints_true)
        pose = solve_pnp(points, keypoints_)

        P = transform(exp_so3(pose.omega), pose.t, points)
        keypoints_pred = projection.compute(P)

        assert_array_almost_equal(t_true, pose.t)
        assert_array_almost_equal(keypoints_true, keypoints_pred)

    case1()
    case2()


X = np.array([
    [0.0, 0.0],
    [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],  # r = 1
    [1.0, np.sqrt(3)]  # r = 2
])


def test_fov_undistort():
    # tan(pi / 6) = 1 / sqrt(3) = 3 / sqrt(3)
    expected = np.array([
        # (pi / 3) / (2 * tan(pi / 6))
        (np.pi / 3) / (2 * np.sqrt(3) / 3),
        # tan(1 * pi / 3) / (2 * 1 * tan(pi / 6))
        np.sqrt(3) / (2 * np.sqrt(3) / 3),
        # tan(2 * pi / 3) / (2 * 2 * tan(pi / 6))
        -np.sqrt(3) / (2 * 2 * np.sqrt(3) / 3)
    ])

    # test 'calc_undistort_factors' separately because
    # we cannot test factors for X[0] using FOV.undistort
    assert_array_almost_equal(undistort_factors(X, omega=np.pi/3), expected)
    assert_array_almost_equal(
        FOV(omega=np.pi/3).undistort(X),
        expected.reshape(-1, 1) * X
    )

    # all factors = 1
    assert_array_almost_equal(FOV(omega=0).undistort(X), X)


def test_fov_distort():
    expected = np.array([
        # (2 * tan(pi / 6)) / (pi / 3)
        (2 * np.sqrt(3) / 3) / (np.pi / 3),
        # arctan(2 * r * tan(pi / 6)) / (pi / 3)
        np.arctan(2 * np.sqrt(3) / 3) / (np.pi / 3),
        # arctan(2 * 2 * tan(pi / 6)) / (pi / 3)
        np.arctan(2 * 2 * np.sqrt(3) / 3) / (np.pi / 3)
    ])

    assert_array_almost_equal(distort_factors(X, omega=np.pi/3), expected)
    assert_array_almost_equal(
        FOV(omega=np.pi/3).distort(X),
        expected.reshape(-1, 1) * X
    )

    assert_array_almost_equal(FOV(omega=0).distort(X), X)
