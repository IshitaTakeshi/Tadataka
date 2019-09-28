from autograd import numpy as np
from vitamine.visual_odometry.pose import Pose


def test_eq():
    R0 = np.identity(3)
    R1 = np.arange(9).reshape(3, 3)
    t0 = np.zeros(3)
    t1 = np.arange(3)

    assert(Pose(R0, t0) == Pose(R0, t0))
    assert(Pose(R1, t1) == Pose(R1, t1))
    assert(Pose(R0, t0) != Pose(R0, t1))
    assert(Pose(R0, t0) != Pose(R1, t0))
    assert(Pose(R0, t0) != Pose(R1, t1))
