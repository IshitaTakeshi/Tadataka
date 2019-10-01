from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from vitamine.pose import Pose
from tests.utils import random_rotation_matrix


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
