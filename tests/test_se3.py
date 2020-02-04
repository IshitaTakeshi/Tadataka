import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.linalg import expm
from tadataka.se3 import exp_se3, log_se3
from tadataka.pose import Pose


def tangent_se3_(xi):
    v1, v2, v3, v4, v5, v6 = xi
    return np.array([
        [0, -v6, v5, v1],
        [v6, 0, -v4, v2],
        [-v5, v4, 0, v3],
        [0, 0, 0, 0]
    ])


def test_exp_se3():
    def run(xi):
        assert_array_almost_equal(
            exp_se3(xi),
            expm(tangent_se3_(xi))
        )

    run(np.array([1, 2, -3, 0, 0, 0]))
    run(np.array([1, 2, -3, 0, 0, 1e-9]))
    run(np.array([1, -1, 2, np.pi / 2, 0, 0]))
    run(np.array([-1, 2, 1, 0, -np.pi / 2, np.pi / 4]))


def test_log_se3():
    def run(xi):
        # test log(exp(xi)) == xi
        assert_array_almost_equal(log_se3(expm(tangent_se3_(xi))), xi)

    run(np.array([1, 2, -3, 0, 0, 0]))
    run(np.array([1, -1, 2, np.pi / 2, 0, 0]))
    run(np.array([-1, 2, 1, 0, -np.pi / 2, np.pi / 4]))


def test_from_se3():
    for i in range(10):
        xi = np.random.uniform(-2, 2, 6)
        pose = Pose.from_se3(xi)
        G = expm(tangent_se3_(xi))
        assert_array_almost_equal(pose.R, G[0:3, 0:3])
        assert_array_almost_equal(pose.t, G[0:3, 3])
