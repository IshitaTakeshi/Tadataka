import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.propagation import propagate_variance


def test_propagation():
    # rotate 90 degrees around the y-axis
    rotation = Rotation.from_rotvec([0, np.pi / 2, 0])
    # forward 2 along the x axis
    t0 = np.array([0, 0, 1])
    t1 = np.array([2, 0, 1])
    # depth1 = depth0 - 2
    pose0 = WorldPose(rotation, t0)
    pose1 = WorldPose(rotation, t1)
    assert_almost_equal(tz01, -2.0)

    uncertaintity = 1.0

    shape = (4, 3)
    depth0 = 4.0
    variance0 = 3.0
    inv_depth_map0 = (1 / depth0) * np.ones(shape)
    variance_map0 = variance0 * np.ones(shape)

    propagate = DepthMapPropagation(tz01, uncertaintity)
    inv_depth_map1, variance_map1 = propagate(inv_depth_map0, variance_map0)

    depth1 = depth0 - 2.0
    assert_array_almost_equal(inv_depth_map1, (1 / depth1) * np.ones(shape))


def test_propagate_variance():
    inv_depths0 = np.array([4.0, 1.0, 2.0])
    inv_depths1 = np.array([2.0, 3.0, 1.0])
    variances0 = np.array([0.5, 1.0, 4.0])

    assert_array_almost_equal(
        propagate_variance(inv_depths0, inv_depths1, variances0, 1.0),
        np.power(inv_depths1 / inv_depths0, 4) * variances0 + 1.0
    )
