from autograd import numpy as np

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
