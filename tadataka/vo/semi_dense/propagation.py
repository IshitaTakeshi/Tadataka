import numpy as np

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def propagate_variance(inv_depths0, inv_depths1, variances0, uncertaintity):
    # we assume few rotation change between timestamp 0 and timestamp 1
    # TODO accept the case rotations significantly change between 0 and 1
    ratio = inv_depths1 / inv_depths0
    print("max ratio", np.max(ratio))
    variances1 = np.power(ratio, 4) * variances0 + uncertaintity
    return variances1


def substitute(array2d, us, values):
    assert(us.shape[0] == values.shape[0])
    xs, ys = us[:, 0], us[:, 1]
    array2d[ys, xs] = values
    return array2d


class Propagation(object):
    def __init__(self, warp2d, uncertaintity=0.01,
                 initial_inv_depth=1.0, initial_variance=1.0):
        self.warp2d = warp2d
        self.uncertaintity = uncertaintity
        self.initial_inv_depth = initial_inv_depth
        self.initial_variance = initial_variance

    def _warp(self, us0, inv_depths0):
        depths0 = invert_depth(inv_depths0)
        us1, depths1 = self.warp2d(us0, depths0)
        inv_depths1 = invert_depth(depths1)
        return np.round(us1).astype(np.int64), inv_depths1

    def __call__(self, inv_depth_map0, variance_map0):
        image_shape = inv_depth_map0.shape

        us0 = image_coordinates(image_shape)
        variances0 = variance_map0.flatten()
        inv_depths0 = inv_depth_map0.flatten()

        us1, inv_depths1 = self._warp(us0, inv_depths0)

        mask = is_in_image_range(us1, image_shape)
        us1 = us1[mask]
        inv_depths0 = inv_depths0[mask]
        inv_depths1 = inv_depths1[mask]
        variances0 = variances0[mask]

        variances1 = propagate_variance(inv_depths0, inv_depths1,
                                        variances0, self.uncertaintity)

        inv_depth_map1 = self.initial_inv_depth * np.ones(image_shape)
        inv_depth_map1 = substitute(inv_depth_map1, us1, inv_depths1)

        variance_map1 = self.initial_variance * np.ones(image_shape)
        variance_map1 = substitute(variance_map1, us1, variances1)

        return inv_depth_map1, variance_map1
