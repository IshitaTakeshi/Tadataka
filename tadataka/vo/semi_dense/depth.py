import numpy as np
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.projection import inv_pi
from tadataka.matrix import get_rotation_translation
from tadataka.triangulation import calc_depth0_
from tadataka.rigid_transform import transform_se3


def calc_ref_inv_depth(T_rk, x_key, inv_depth_key):
    depth_key = invert_depth(inv_depth_key)
    p_key = inv_pi(x_key, depth_key)
    p_ref = transform_se3(T_rk, p_key)
    depth_ref = p_ref[2]
    return invert_depth(depth_ref)


def calc_key_depth(T_rk, x_key, x_ref):
    R_rk, t_rk = get_rotation_translation(T_rk)
    return calc_depth0_(R_rk, t_rk, x_key, x_ref)


# TODO move this to a proper file
def clamp(value, min_, max_):
    return min(max(value, min_), max_)


class InvDepthSearchRange(object):
    def __init__(self, min_inv_depth, max_inv_depth, factor=2.0):
        assert(0 < min_inv_depth < max_inv_depth)
        self.factor = factor
        self.min_inv_depth = min_inv_depth
        self.max_inv_depth = max_inv_depth

    def _clamp(self, value):
        return clamp(value, self.min_inv_depth, self.max_inv_depth)

    def __call__(self, hypothesis):
        inv_depth, variance = hypothesis
        assert(variance >= 0.0)
        min_ = inv_depth - self.factor * variance
        max_ = inv_depth + self.factor * variance

        if max_ <= self.min_inv_depth or self.max_inv_depth <= min_:
            return None

        min_inv_depth = self._clamp(min_)
        max_inv_depth = self._clamp(max_)
        return min_inv_depth, max_inv_depth


def depth_search_range(min_inv_depth, max_inv_depth):
    min_depth = invert_depth(max_inv_depth)
    max_depth = invert_depth(min_inv_depth)
    return min_depth, max_depth
