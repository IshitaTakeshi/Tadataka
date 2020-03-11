import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.propagation import propagate_variance


def test_propagation():
    # rotate 90 degrees around the y-axis
    rotation = Rotation.from_rotvec([0, np.pi / 2, 0])

    t0 = np.array([0, 0, 1])
    t1 = np.array([2, 0, 1])

    pose0 = WorldPose(rotation, t0)
    pose1 = WorldPose(rotation, t1)

    depths0 = np.array([4.0, 5.0])
    us0 = np.array([
        [3.0, 2.0],
        [-1.0, 4.0]
    ])

    warp = Warp3D(pose0, pose1)
    us1, inv_depths1 = propagate(us0, inv_depths0)
    assert_array_almost_equal(inv_depths1, )


def test_propagate_variance():
    inv_depths0 = np.array([4.0, 1.0, 2.0])
    inv_depths1 = np.array([2.0, 3.0, 1.0])
    variances0 = np.array([0.5, 1.0, 4.0])

    assert_array_almost_equal(
        propagate_variance(inv_depths0, inv_depths1, variances0, 1.0),
        np.power(inv_depths1 / inv_depths0, 4) * variances0 + 1.0
    )
