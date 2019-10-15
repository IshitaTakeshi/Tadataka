import pytest

from autograd import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from vitamine.so3 import rodrigues, exp_so3

from vitamine.pose import Pose, solve_pnp
from vitamine.exceptions import NotEnoughInliersException
from vitamine.camera import CameraParameters
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.observations import generate_translations
from vitamine.rigid_transform import transform

from tests.utils import random_rotation_matrix


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

points = np.array([
   [4, -1, 3],
   [1, -3, -2],
   [-2, 3, -2],
   [-3, -2, -5],
   [4, 1, 1],
   [-2, 3, 1],
   [-4, 4, -1]
])

omegas = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.2, 0.0, 0.1],
    [3.2, 1.1, 1.0],
    [3.1, -0.2, 0.0],
    [-1.0, 0.2, 3.1]
])

translations = generate_translations(rodrigues(omegas), points)


def test_solve_pnp():
    for omega_true, t_true in zip(omegas, translations):
        P = transform(exp_so3(omega_true), t_true, points)
        keypoints_true = projection.compute(P)

        pose = solve_pnp(points, keypoints_true)

        P = transform(exp_so3(pose.omega), pose.t, points)
        keypoints_pred = projection.compute(P)

        # omega_true and omega_pred can be different but
        # they have to be linearly dependent and satisfy
        # norm(omega_true) - norm(omega_pred) = 2 * n * pi

        assert_array_almost_equal(t_true, pose.t)
        assert_array_almost_equal(keypoints_true, keypoints_pred)

    P = transform(exp_so3(omegas[0]), translations[0], points)
    keypoints0 = projection.compute(P)

    # 6 correspondences
    # this should be able to run
    solve_pnp(points[0:6], keypoints0[0:6])

    with pytest.raises(NotEnoughInliersException):
        # not enough correspondences
        solve_pnp(points[0:5], keypoints0[0:5])


def test_eq():
    omega0 = np.zeros(3)
    omega1 = np.arange(3)
    t0 = np.zeros(3)
    t1 = np.arange(3)

    assert(Pose(omega0, t0) == Pose(omega0, t0))
    assert(Pose(omega1, t1) == Pose(omega1, t1))
    assert(Pose(omega0, t0) != Pose(omega0, t1))
    assert(Pose(omega0, t0) != Pose(omega1, t0))
    assert(Pose(omega0, t0) != Pose(omega1, t1))

    R0 = random_rotation_matrix(3)
    R1 = random_rotation_matrix(3)
    t0 = np.zeros(3)
    t1 = np.arange(3)

    assert(Pose(R0, t0) == Pose(R0, t0))
    assert(Pose(R1, t1) == Pose(R1, t1))
    assert(Pose(R0, t0) != Pose(R0, t1))
    assert(Pose(R0, t0) != Pose(R1, t0))
    assert(Pose(R0, t0) != Pose(R1, t1))


def test_identity():
    pose = Pose.identity()
    assert_array_equal(pose.omega, np.zeros(3))
    assert_array_equal(pose.t, np.zeros(3))


def test_R():
    pose = Pose(np.zeros(3), np.zeros(3))
    assert_array_almost_equal(pose.R, np.identity(3))

    pose = Pose(np.array([np.pi, 0, 0]), np.zeros(3))
    assert_array_almost_equal(pose.R, np.diag([1, -1, -1]))
