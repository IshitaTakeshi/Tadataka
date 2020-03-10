import numpy as np

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.common import invert_depth


def propagate_variance(inv_depths0, inv_depths1, variances0, uncertaintity):
    # we assume few rotation change between timestamp 0 and timestamp 1
    # TODO accept the case rotations significantly change between 0 and 1
    ratio = inv_depths1 / inv_depths0
    variances1 = np.power(ratio, 4) * variances0 + uncertaintity
    return variances1


class DepthMapPropagation(object):
    def __init__(self, tz01, uncertaintity):
        self.tz = tz01
        self.uncertaintity = uncertaintity

    def __call__(self, inv_depth_map0, variance_map0):
        depth_map0 = invert_depth(inv_depth_map0)
        depth_map1 = depth_map0 + self.tz
        inv_depth_map1 = invert_depth(depth_map1)

        ratio = inv_depth_map1 / inv_depth_map0
        variance_map1 = np.power(ratio, 4) * variance_map0 + self.uncertaintity
        return inv_depth_map1, variance_map1
