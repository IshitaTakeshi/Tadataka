import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.propagation import DepthMapPropagation, calc_depth_offset


def test_calc_depth_offset():
    # rotate 90 degrees around the y-axis
    rotation = Rotation.from_rotvec([0, np.pi / 2, 0])
    # forward 2 along the x axis
    t0 = np.array([0, 0, 1])
    t1 = np.array([2, 0, 1])
    # depth1 = depth0 - 2
    pose0 = WorldPose(rotation, t0)
    pose1 = WorldPose(rotation, t1)
    tz01 = calc_depth_offset(pose0, pose1)
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

    variance1 = pow((1 / depth1) / (1 / depth0), 4) * variance0 + 1.0
    assert_array_almost_equal(variance_map1, variance1 * np.ones(shape))
